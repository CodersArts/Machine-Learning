#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import pandas as pd
import fitz  # PyMuPDF

output_DIR = os.path.join(os.getcwd(), 'output')


def is_pdf_by_extension(file_path : str):
    """
    Check if a file is a PDF based on its file extension.

    Args:
        file_path (str): The path to the file.

    Returns:
        bool: True if the file has a '.pdf' extension, False otherwise.
    """
    # Get the file extension from the file path
    file_extension = os.path.splitext(file_path)[1].lower()

    # Check if the file extension is '.pdf'
    return file_extension == '.pdf'




def num_page_check(pdf_path :str, N :int = 10):
    """
    Check if a PDF has more than 'N' pages.

    Args:
        pdf_path (str): The path to the PDF file.

    Returns:
        bool: True if the PDF has more than N pages, False otherwise.
    """
    
    pdf_document = fitz.open(pdf_path)
    num_pages = len(pdf_document)

    return num_pages > N
    
    

def split_pdf(input_pdf_path : str, pages_per_split : int = 10):
    
    file_name = input_pdf_path.split('\\')[-1].split('.pdf')[0]

    # Create a directory to store the split PDFs
    output_dir = 'split_pdfs'
    os.makedirs(output_dir, exist_ok=True)

    # Open the input PDF file
    pdf_document = fitz.open(input_pdf_path)

    # Calculate the number of splits needed
    num_pages = len(pdf_document)
    num_splits = (num_pages + pages_per_split - 1) // pages_per_split

    for split_index in range(num_splits):
        # Create a new PDF document for each split
        pdf_split = fitz.open()

        # Determine the range of pages for this split
        start_page = split_index * pages_per_split
        end_page = min(start_page + pages_per_split, num_pages)

        # Copy pages to the new PDF document
        for page_num in range(start_page, end_page):
            page = pdf_document.load_page(page_num)
            pdf_split.insert_pdf(pdf_document, from_page=page_num, to_page=page_num)

        # Save the split PDF to a new file
        output_pdf_path = os.path.join(output_dir, f'{file_name}_split_{split_index + 1}.pdf')
        pdf_split.save(output_pdf_path)
        pdf_split.close()

        print(f'Saved {output_pdf_path} with pages {start_page + 1}-{end_page}')

    # Close the input PDF document
    pdf_document.close()

    print('Splitting complete.')
    

    
def combine_excels(excel_files : list, file_name : str):
    
    """
    Function to combine multiple excel files into a single excel file
    """
    
    # Name of the output Excel file
    output_excel_file = os.path.join( output_DIR,  f'{file_name}_combined.xlsx' )

    # Create an ExcelWriter object
    excel_writer = pd.ExcelWriter(output_excel_file, engine='xlsxwriter')

    # Create a Pandas Excel writer using XlsxWriter as the engine
    with pd.ExcelWriter(output_excel_file, engine='xlsxwriter') as excel_writer:
        # Iterate through the list of Excel files
        for file_index, excel_file in enumerate(excel_files):
            # Read all sheets from the current Excel file into a dictionary
            xls = pd.read_excel(excel_file, sheet_name=None)

            # Iterate through the sheets in the dictionary and write them to the output Excel file
            for sheet_name, df in xls.items():
                unique_sheet_name = f'{sheet_name}_f{file_index + 1}'
                df.to_excel(excel_writer, sheet_name= unique_sheet_name, index=False)

    print(f'Sheets from {len(excel_files)} Excel files combined into {output_excel_file}.')


    
def delete_files(directory : str):
    """
    Delete all files in a directory.

    Args:
        directory (str): The path to the directory containing the files to be deleted.
    """
    try:
        # List all files in the directory
        files = os.listdir(directory)

        # Iterate through the files and delete each one
        for file in files:
            file_path = os.path.join(directory, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f'Deleted: {file_path}')
        
        print(f'All files in {directory} have been deleted.')

    except Exception as e:
        print(f'An error occurred: {e}')

