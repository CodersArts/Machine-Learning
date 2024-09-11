#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import pandas as pd
import json
import re
import time

from google.api_core.client_options import ClientOptions 
from google.cloud import translate_v2 as translate
from google.cloud import documentai_v1 as documentai

from typing import List, Sequence

excel_output_DIR = os.path.join(os.getcwd(), 'excel files')

# Function to translate text using Google Cloud Translation API
def translate_text(text, target_language = 'en'):
    '''
    function to translate from Marathi to English
    '''
    translate_client = translate.Client()
    result = translate_client.translate(text, target_language =  target_language, source_language = 'mr')
    return result['translatedText']


def quickstart(
    project_id: str,
    location: str,
    file_path: str,
    processor_display_name: str = "My Processor",
):
    # You must set the `api_endpoint`if you use a location other than "us".
    opts = ClientOptions(api_endpoint=f"{location}-documentai.googleapis.com")

    client = documentai.DocumentProcessorServiceClient(client_options=opts)

    # The full resource name of the location, e.g.:
    # `projects/{project_id}/locations/{location}`
    parent = client.common_location_path(project_id, location)

    # Read the file into memory
    with open(file_path, "rb") as image:
        image_content = image.read()

    # Load binary data
    raw_document = documentai.RawDocument(
        content=image_content,
        mime_type="application/pdf",  # Refer to https://cloud.google.com/document-ai/docs/file-types for supported file types
    )

    # Configure the process request
    # `processor.name` is the full resource name of the processor, e.g.:
    # `projects/{project_id}/locations/{location}/processors/{processor_id}`
    request = documentai.ProcessRequest(name=processor_display_name, raw_document=raw_document)

    result = client.process_document(request=request)

    # For a full list of `Document` object attributes, reference this page:
    # https://cloud.google.com/document-ai/docs/reference/rest/v1/Document
    document = result.document
    #table = result.table

    # Read the text recognition output from the processor
    #print("The document contains the following text:")
    #print(document.text)
    #print(table)
    
    return document



def layout_to_text(layout: documentai.Document.Page.Layout, text: str) -> str:
    """
    Document AI identifies text in different parts of the document by their
    offsets in the entirety of the document's text. This function converts
    offsets to a string.
    """
    response = ""
    # If a text segment spans several lines, it will
    # be stored in different text segments.
    for segment in layout.text_anchor.text_segments:
        start_index = int(segment.start_index)
        end_index = int(segment.end_index)
        response += text[start_index:end_index]
    return response
        

def text_anchor_to_text(text_anchor: documentai.Document.TextAnchor, text: str) -> str:
    """
    Document AI identifies table data by their offsets in the entirity of the
    document's text. This function converts offsets to a string.
    """
    response = ""
    # If a text segment spans several lines, it will
    # be stored in different text segments.
    for segment in text_anchor.text_segments:
        start_index = int(segment.start_index)
        end_index = int(segment.end_index)
        response += text[start_index:end_index]
    return response.strip().replace("\n", " ")


def get_table_data(rows: Sequence[documentai.Document.Page.Table.TableRow], text: str) -> List[List[str]]:
        """
        Get Text data from table rows
        """
        all_values: List[List[str]] = []
        for row in rows:
            current_row_values: List[str] = []
            for cell in row.cells:
                current_row_values.append(
                    text_anchor_to_text(cell.layout.text_anchor, text)
                )
            all_values.append(current_row_values)
        return all_values
    
    
    
def print_table_rows(table_rows: Sequence[documentai.Document.Page.Table.TableRow], text: str) -> None:
    for table_row in table_rows:
        row_text = ""
        for cell in table_row.cells:
            cell_text = layout_to_text(cell.layout, text)
            row_text += f"{repr(cell_text.strip())} | "
        print(row_text)
        
        

        
def process_pdf( document : documentai.Document, file_name : str ):
    '''
    This function takes in a google cloud document, reads its tabular layout and saves the content of the table in an excel file
    
    parameter              type
    document               google.cloud.documentai_v1.types.document.Document
    file_name              name of the pdf file (without the extension)
     
    '''
    
    text = document.text

    # Read the form fields and tables output from the processor
    ctr = 0
    json_dict = {}

    for page in document.pages:
        
        ctr += 1
        start_time = time.time()
        
        print(f"\n\n**** Page {page.page_number} ****")

        print(f"\nFound {len(page.tables)} table(s):")
        tab_ctr=0
        for table in page.tables:
            
            tab_ctr+=1
            num_columns = len(table.header_rows[0].cells)
            num_rows = len(table.body_rows)
            print(f"Table with {num_columns} columns and {num_rows} rows:")

            # Print header rows
            #print("Columns:")
            #print_table_rows(table.header_rows, text)
            # Print body rows
            #print("Table body data:")
            #print_table_rows(table.body_rows, text)


            header_row_values: List[List[str]] = []
            body_row_values: List[List[str]] = []
            
            extract_start = time.time()
            
            header_row_values = get_table_data(table.header_rows, document.text)
            body_row_values = get_table_data(table.body_rows, document.text)
            
            extract_end = time.time()
            
            

            # translate the text before adding to dataframe
            #translate_start = time.time()
            #translated_header = [translate_text(re.sub("|", "" , i)) for i in header_row_values[0]]
            #translated_body = [[ translate_text(re.sub("|", "" , ele)) for ele in row] for row in body_row_values ]
            #print(f'translated page : {ctr}')
            #translate_end = time.time()

            # add the text to the dataframe
            df = pd.DataFrame(
            data = body_row_values,
            columns = header_row_values
            )
            
            if ctr == 1:
                

                df.to_excel(os.path.join(excel_output_DIR, f'{file_name}.xlsx'), sheet_name = str(ctr))
                print(f'Processed page: {ctr}')
                print('-------------------------------------------------')
                end_time = time.time()
                
            else:
                with pd.ExcelWriter(os.path.join(excel_output_DIR, f'{file_name}.xlsx'), engine='openpyxl', mode='a', if_sheet_exists='new') as writer:
                # Write your DataFrame to a new sheet in the existing Excel file
                    df.to_excel(writer, sheet_name = f'page {str(ctr)}')
                    
                print(f'Processed page: {ctr}')
                print('-------------------------------------------------')
                end_time = time.time()
                
                
            elapsed_time_total = end_time - start_time
            print(f'Time taken to process page {ctr} : {elapsed_time_total/60} min')
            
            print('Breakdown of time:')
            elapsed_time_extraction = extract_end - extract_start
            print(f'Time taken to extract tables : {elapsed_time_extraction} s')
            
            #elapsed_time_translation = translate_end - translate_start
            #print(f'Time taken to translate text : {elapsed_time_translation/60} min')
            
    print('Current pdf has been processed')

