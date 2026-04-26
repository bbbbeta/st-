import pandas as pd
import numpy as np
import os
from openai import OpenAI
import json
import time
import re
from typing import List, Dict, Tuple
import math

class LLMGeneSelector:
    """
    Use LLM to select tissue-relevant genes by processing ALL candidates in batches.
    """

    def __init__(self, api_key: str = None, model: str = "gpt-5", base_url: str = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.base_url = base_url
        if base_url:
            self.client = OpenAI(api_key=self.api_key, base_url=base_url)
        else:
            self.client = OpenAI(api_key=self.api_key)

    def load_gene_knowledge(self, knowledge_path: str) -> pd.DataFrame:
        """
        Load, clean, and format gene knowledge based on specific column names.
        """
        print(f"Loading gene knowledge from {knowledge_path}...")
        try:
            df = pd.read_csv(knowledge_path, quotechar='"', skipinitialspace=True)
        except Exception as e:
            print(f"Error reading CSV: {e}")
            return pd.DataFrame()

        full_name_col = None
        if 'description_chicken' in df.columns:
            full_name_col = 'description_chicken'
        elif 'target_description' in df.columns:
            full_name_col = 'target_description'

        if 'gene' not in df.columns:
            raise ValueError(f"Missing required column 'gene' in {knowledge_path}")
        if 'gene_summary' not in df.columns:
            df['gene_summary'] = ''

        df['gene_summary'] = df['gene_summary'].fillna('')
        if full_name_col:
            df[full_name_col] = df[full_name_col].fillna('')

        def clean_and_combine(row):
            summary = str(row['gene_summary']).strip()

            full_name = ""
            if full_name_col:
                full_name = str(row[full_name_col]).strip()

            if summary.lower() in ['none', 'null', 'nan']:
                summary = ""
            if full_name.lower() in ['none', 'null', 'nan']:
                full_name = ""

            if summary:
                summary = re.sub(r'\s*\[provided by.*?\]', '', summary)

            final_desc = ""

            if summary:
                if full_name and full_name not in summary:
                    final_desc = f"{full_name}. {summary}"
                else:
                    final_desc = summary
            elif full_name:
                final_desc = f"Full Name: {full_name}"

            return final_desc

        df['combined_desc'] = df.apply(clean_and_combine, axis=1)

        original_count = len(df)
        df = df[df['combined_desc'].notna() & (df['combined_desc'] != '')]

        print(f"Knowledge Base: Loaded {len(df)} valid genes (filtered out {original_count - len(df)} empty records).")

        if len(df) > 0:
            print("Sample cleaned descriptions:")
            for _, row in df.head(3).iterrows():
                print(f" -> {row['gene']}: {row['combined_desc'][:100]}...")

        return df

    def _parse_response(self, content: str) -> Dict:
        """Helper to safely extract JSON from LLM response"""
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
            if match:
                return json.loads(match.group(1))
            raise ValueError("No valid JSON found in response")

    def construct_batch_prompt(self, tissue_type: str,
                               batch_genes: pd.DataFrame,
                               batch_index: int,
                               total_batches: int) -> Tuple[str, str]:
        """Construct prompt using the cleaned 'combined_desc'."""
        gene_info_list = []
        for _, row in batch_genes.iterrows():
            desc = row['combined_desc']
            if len(desc) > 250:
                desc = desc[:247] + "..."
            gene_info_list.append(f"- {row['gene']}: {desc}")

        gene_info_str = "\n".join(gene_info_list)

        system_prompt = f"""You are an expert in bioinformatics.
You are currently processing batch {batch_index}/{total_batches} of a gene screening task for {tissue_type} tissue.
Your goal is to identify genes relevant for spatial domain identification, cell type specification, or regional identity.
"""

        user_prompt = f"""Here is a list of candidate genes and their cleaned functional descriptions:

{gene_info_str}

Task:
Analyze the functions of these genes. Select ONLY the ones that are highly relevant to {tissue_type} structure, function, or cell types.
- Prioritize genes with known functions in {tissue_type}.
- Note that 'long intergenic non-protein coding RNA' (LINC) or similar full names can imply regulatory roles even without a detailed summary.

Format requirements:
- Return valid JSON.
- JSON key: "selected_genes" (list of strings).
- JSON key: "reasoning" (brief summary of this batch).

JSON Example:
{{
    "selected_genes": ["GeneA", "GeneB"],
    "reasoning": "Selected GeneA for neuronal function..."
}}"""
        return system_prompt, user_prompt

    def select_genes_with_llm(self,
                             tissue_type: str,
                             gene_list: List[str],
                             knowledge_path: str,
                             tokens_per_gene_est: int = 80,
                             max_context_tokens: int = 200000,
                             max_retries: int = 3) -> Tuple[List[str], str]:
        """Main execution engine for LLM-based gene selection."""
        print(f"Loading gene knowledge from {knowledge_path}...")
        gene_knowledge = self.load_gene_knowledge(knowledge_path)

        valid_knowledge = gene_knowledge[gene_knowledge['gene'].isin(gene_list)]
        genes_with_data = valid_knowledge['gene'].tolist()

        if len(genes_with_data) == 0:
            return [], "No gene knowledge available."

        batch_size = int(max_context_tokens / tokens_per_gene_est)

        total_genes = len(genes_with_data)
        total_batches = math.ceil(total_genes / batch_size)

        print(f"Strategy: Processing {total_genes} genes in {total_batches} batches (Batch size: ~{batch_size}).")

        all_selected_genes = []
        batch_reasonings = []

        for i in range(total_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, total_genes)
            current_batch_df = valid_knowledge.iloc[start_idx:end_idx]

            print(f"\n{'='*20} Processing Batch {i+1}/{total_batches} {'='*20}")

            system_prompt, user_prompt = self.construct_batch_prompt(
                tissue_type, current_batch_df, i+1, total_batches
            )

            print(f"[DEBUG] Prompt Preview (First 300 chars of user prompt):")
            start_marker = "descriptions:\n\n"
            start_pos = user_prompt.find(start_marker)
            if start_pos != -1:
                print(user_prompt[start_pos + len(start_marker):][:300] + "...")
            else:
                print(user_prompt[:300] + "...")

            batch_success = False
            for attempt in range(max_retries):
                try:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        temperature=0.3,
                        response_format={"type": "json_object"}
                    )

                    content = response.choices[0].message.content
                    result = self._parse_response(content)

                    batch_selected = result.get("selected_genes", [])
                    current_batch_genes = set(current_batch_df['gene'].tolist())
                    valid_batch_selected = [g for g in batch_selected if g in current_batch_genes]

                    print(f"   > Batch Result: Selected {len(valid_batch_selected)} / {len(current_batch_genes)}")

                    all_selected_genes.extend(valid_batch_selected)
                    batch_reasonings.append(f"Batch {i+1}: {result.get('reasoning', '')}")
                    batch_success = True
                    break

                except Exception as e:
                    print(f"   > Error (Attempt {attempt+1}): {e}")
                    time.sleep(1)

            if not batch_success:
                print(f"   > FAILED Batch {i+1}. Skipping.")

        unique_selected_genes = list(set(all_selected_genes))
        final_reasoning = "\n".join(batch_reasonings)

        return unique_selected_genes, final_reasoning

    def save_selected_genes(self, selected_genes: List[str], reasoning: str, output_path: str):
        result = {
            "selected_genes": selected_genes,
            "num_genes": len(selected_genes),
            "reasoning": reasoning
        }
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Saved results to {output_path}")
