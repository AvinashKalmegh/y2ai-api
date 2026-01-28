"""Patch breadth_dial.py to add pagination"""
import re

with open('dials/breadth_dial.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find and replace the method using regex
pattern = r'def _fetch_from_supabase\(self, tickers: List\[str\], days: int\) -> Optional\[pd\.DataFrame\]:.*?return None\n'

replacement = '''def _fetch_from_supabase(self, tickers: List[str], days: int) -> Optional[pd.DataFrame]:
        """Fetch from Supabase with pagination."""
        try:
            start_date = (datetime.now() - timedelta(days=days * 2)).strftime("%Y-%m-%d")
            all_data = []
            offset = 0
            batch_size = 1000
            while True:
                response = self.supabase.table("price_history") \\
                    .select("date, ticker, close") \\
                    .in_("ticker", tickers) \\
                    .gte("date", start_date) \\
                    .order("date", desc=True) \\
                    .range(offset, offset + batch_size - 1) \\
                    .execute()
                if not response.data:
                    break
                all_data.extend(response.data)
                if len(response.data) < batch_size:
                    break
                offset += batch_size
            if all_data:
                df = pd.DataFrame(all_data)
                df["date"] = pd.to_datetime(df["date"])
                logger.info(f"Fetched {len(df)} rows via pagination")
                return df
        except Exception as e:
            logger.warning(f"Supabase fetch failed: {e}")
        return None
'''

new_content = re.sub(pattern, replacement, content, flags=re.DOTALL)

if new_content != content:
    with open('dials/breadth_dial.py', 'w', encoding='utf-8') as f:
        f.write(new_content)
    print("Patched successfully!")
else:
    print("Pattern not found")