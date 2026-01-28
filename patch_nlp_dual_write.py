"""
Patch NCI, NPD, Burst, EVI files to write to both sheets.
"""

def patch_file(filename, sheet_name):
    """Add dual-write function and update append_row calls."""
    
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Skip if already patched
    if 'def write_to_both_sheets' in content:
        print(f'{filename}: Already patched')
        return
    
    # Add the helper function after SPREADSHEET_NAME_2
    helper_function = '''

def write_to_both_sheets(gc, sheet_name, row_data, is_header=False):
    """Write row to both spreadsheets."""
    # Primary sheet
    try:
        sheet1 = gc.open(SPREADSHEET_NAME).worksheet(sheet_name)
        sheet1.append_row(row_data, value_input_option='USER_ENTERED')
    except Exception as e:
        print(f"Error writing to primary sheet: {e}")
    
    # Secondary sheet (Vikram-Develop-This)
    try:
        sheet2 = gc.open(SPREADSHEET_NAME_2).worksheet(sheet_name)
        sheet2.append_row(row_data, value_input_option='USER_ENTERED')
    except Exception as e:
        print(f"Error writing to secondary sheet: {e}")
'''
    
    # Insert helper function after SPREADSHEET_NAME_2 line
    insert_marker = "SPREADSHEET_NAME_2 = 'Vikram-Develop-This'"
    if insert_marker in content:
        # Find end of that line
        idx = content.find(insert_marker)
        end_of_line = content.find('\n', idx)
        # Find next line that's not a comment
        next_line_end = content.find('\n', end_of_line + 1)
        
        content = content[:next_line_end] + helper_function + content[next_line_end:]
    
    # Now replace sheet.append_row calls
    # Pattern: sheet.append_row(something)
    # Replace with: write_to_both_sheets(gc, 'SHEET_NAME', something)
    
    lines = content.split('\n')
    new_lines = []
    
    for line in lines:
        if 'sheet.append_row(' in line and 'def ' not in line:
            # Extract the argument
            indent = len(line) - len(line.lstrip())
            spaces = ' ' * indent
            
            # Get what's inside append_row()
            start = line.find('append_row(') + len('append_row(')
            end = line.rfind(')')
            arg = line[start:end]
            
            # Check if it has value_input_option
            if 'value_input_option' in arg:
                # Remove the value_input_option part
                arg = arg.split(',')[0].strip()
            
            new_line = f"{spaces}write_to_both_sheets(gc, '{sheet_name}', {arg})"
            new_lines.append(new_line)
        else:
            new_lines.append(line)
    
    content = '\n'.join(new_lines)
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f'{filename}: Patched successfully')


def main():
    files_and_sheets = [
        ('nci_narrative_coherence.py', 'NCI_Dial'),
        ('npd_narrative_polarity_drift.py', 'NPD_Dial'),
        ('burst_keyword_detection.py', 'Burst_Dial'),
        ('evi_event_volatility_index.py', 'EVI_Dial'),
    ]
    
    print("Patching files for dual-sheet writing...\n")
    
    for filename, sheet_name in files_and_sheets:
        try:
            patch_file(filename, sheet_name)
        except Exception as e:
            print(f'{filename}: Error - {e}')
    
    print("\nDone! Both sheets will now be updated when these scripts run.")


if __name__ == "__main__":
    main()