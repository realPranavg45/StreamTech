import pandas as pd
import numpy as np
import json
import xml.etree.ElementTree as ET
import xml.sax
import xml.dom.minidom
import requests
from typing import Dict, List, Optional, Union, Any, Tuple
from io import StringIO, BytesIO
import re
from datetime import datetime
import warnings

class DataIngestionBackend:
    """
    A comprehensive backend for ingesting files and API data into pandas DataFrames.
    Supports CSV, XLSX, JSON, XML files and API endpoints with various processing options.
    """
    
    def __init__(self):
        self.supported_file_types = ['csv', 'xlsx', 'xls', 'json', 'xml']
        self.xml_parsing_methods = [
            'elementtree',
            'dom', 
            'sax',
            'iterparse',
            'xpath'
        ]
    
    def detect_date_columns(self, df: pd.DataFrame, sample_size: int = 100) -> List[str]:
        """
        Detect columns that likely contain date information.
        
        Args:
            df: Input DataFrame
            sample_size: Number of rows to sample for detection
            
        Returns:
            List of column names that appear to contain dates
        """
        if df is None or df.empty:
            return []
            
        date_columns = []
        
        # Common date patterns
        date_patterns = [
            r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',  # dd/mm/yyyy, mm/dd/yyyy
            r'\d{2,4}[/-]\d{1,2}[/-]\d{1,2}',  # yyyy/mm/dd
            r'\d{1,2}-\w{3}-\d{2,4}',          # dd-Jan-yyyy
            r'\w{3}\s+\d{1,2},?\s+\d{2,4}',    # Jan 01, 2023
            r'\d{2,4}-\d{2}-\d{2}',            # yyyy-mm-dd (ISO format)
            r'\d{2,4}-\d{1,2}-\d{1,2}',        # yyyy-m-d (ISO format variants)
        ]
        
        # Date keywords to look for in column names
        date_keywords = ['date', 'time', 'created', 'updated', 'modified', 'timestamp', 
                        'birth', 'dob', 'expire', 'start', 'end', 'due', 'schedule']
        
        for col in df.columns:
            try:
                # Skip if column is already datetime
                if pd.api.types.is_datetime64_any_dtype(df[col]):
                    date_columns.append(col)
                    continue
                
                # Check column name for date keywords
                col_name_lower = str(col).lower()
                if any(keyword in col_name_lower for keyword in date_keywords):
                    date_columns.append(col)
                    continue
                
                # Check if column contains object/string data
                if df[col].dtype == 'object' or pd.api.types.is_string_dtype(df[col]):
                    # Sample non-null values
                    non_null_values = df[col].dropna()
                    if len(non_null_values) == 0:
                        continue
                    
                    sample_values = non_null_values.head(min(sample_size, len(non_null_values))).astype(str)
                    
                    # Check if values match date patterns
                    date_like_count = 0
                    for value in sample_values:
                        value_str = str(value).strip()
                        if not value_str or value_str.lower() in ['nan', 'none', 'null', '']:
                            continue
                            
                        for pattern in date_patterns:
                            if re.search(pattern, value_str):
                                date_like_count += 1
                                break
                        
                        # Also try pandas to_datetime to see if it can parse
                        try:
                            pd.to_datetime(value_str, errors='raise')
                            date_like_count += 1
                        except:
                            pass
                    
                    # If more than 30% of sampled values look like dates
                    if len(sample_values) > 0 and date_like_count / len(sample_values) > 0.3:
                        date_columns.append(col)
                
                # Check numeric columns that might be timestamps
                elif pd.api.types.is_numeric_dtype(df[col]):
                    # Check if values could be unix timestamps
                    non_null_values = df[col].dropna()
                    if len(non_null_values) > 0:
                        sample_values = non_null_values.head(min(sample_size, len(non_null_values)))
                        
                        # Check for unix timestamp patterns
                        timestamp_like = 0
                        for value in sample_values:
                            try:
                                # Unix timestamp (seconds) range check
                                if 946684800 <= float(value) <= 2147483647:  # 2000 to 2038
                                    timestamp_like += 1
                                # Unix timestamp (milliseconds) range check
                                elif 946684800000 <= float(value) <= 2147483647000:
                                    timestamp_like += 1
                            except:
                                pass
                        
                        if len(sample_values) > 0 and timestamp_like / len(sample_values) > 0.5:
                            date_columns.append(col)
                            
            except Exception as e:
                # Skip problematic columns but continue processing
                warnings.warn(f"Error detecting dates in column '{col}': {str(e)}")
                continue
        
        # Remove duplicates while preserving order
        seen = set()
        unique_date_columns = []
        for col in date_columns:
            if col not in seen:
                seen.add(col)
                unique_date_columns.append(col)
                
        return unique_date_columns
    
    def convert_date_columns(self, df: pd.DataFrame, date_columns: List[str], 
                           target_format: str = 'dd/mm/yyyy', force_conversion: bool = True) -> pd.DataFrame:
        """
        Convert specified columns to datetime and format them.
        
        Args:
            df: Input DataFrame
            date_columns: List of column names to convert
            target_format: Target date format (currently only supports dd/mm/yyyy)
            force_conversion: If True, try multiple parsing methods aggressively
            
        Returns:
            DataFrame with converted date columns
        """
        if df is None or df.empty:
            return df
            
        df_copy = df.copy()
        
        for col in date_columns:
            if col not in df_copy.columns:
                warnings.warn(f"Column '{col}' not found in DataFrame")
                continue
                
            try:
                original_col = df_copy[col].copy()
                
                # Skip if already in correct format
                if df_copy[col].dtype == 'object':
                    # Check if already in dd/mm/yyyy format
                    sample_values = df_copy[col].dropna().head(10)
                    if len(sample_values) > 0:
                        dd_mm_yyyy_pattern = r'^\d{1,2}/\d{1,2}/\d{4}$'
                        already_formatted = sum(1 for val in sample_values 
                                              if isinstance(val, str) and re.match(dd_mm_yyyy_pattern, val))
                        if already_formatted / len(sample_values) > 0.8:
                            continue
                
                # Method 1: Direct pandas to_datetime with various formats
                datetime_col = None
                parsing_methods = [
                    {'dayfirst': True, 'errors': 'coerce'},
                    {'dayfirst': False, 'errors': 'coerce'},
                    {'format': '%d/%m/%Y', 'errors': 'coerce'},
                    {'format': '%m/%d/%Y', 'errors': 'coerce'},
                    {'format': '%Y/%m/%d', 'errors': 'coerce'},
                    {'format': '%d-%m-%Y', 'errors': 'coerce'},
                    {'format': '%m-%d-%Y', 'errors': 'coerce'},
                    {'format': '%Y-%m-%d', 'errors': 'coerce'},
                    {'format': '%d.%m.%Y', 'errors': 'coerce'},
                    {'format': '%Y%m%d', 'errors': 'coerce'},
                ]
                
                for method in parsing_methods:
                    try:
                        temp_datetime = pd.to_datetime(original_col, **method)
                        valid_count = temp_datetime.notna().sum()
                        if valid_count > 0:
                            if datetime_col is None or valid_count > datetime_col.notna().sum():
                                datetime_col = temp_datetime
                    except:
                        continue
                
                # Method 2: Handle numeric timestamps
                if datetime_col is None or datetime_col.notna().sum() == 0:
                    if pd.api.types.is_numeric_dtype(original_col):
                        try:
                            # Try unix timestamp (seconds)
                            temp_datetime = pd.to_datetime(original_col, unit='s', errors='coerce')
                            if temp_datetime.notna().sum() > 0:
                                datetime_col = temp_datetime
                            else:
                                # Try unix timestamp (milliseconds)
                                temp_datetime = pd.to_datetime(original_col, unit='ms', errors='coerce')
                                if temp_datetime.notna().sum() > 0:
                                    datetime_col = temp_datetime
                        except:
                            pass
                
                # Method 3: Custom parsing for problematic formats
                if (datetime_col is None or datetime_col.notna().sum() == 0) and force_conversion:
                    try:
                        custom_parsed = []
                        for value in original_col:
                            if pd.isna(value):
                                custom_parsed.append(pd.NaT)
                                continue
                                
                            value_str = str(value).strip()
                            if not value_str or value_str.lower() in ['nan', 'none', 'null', '']:
                                custom_parsed.append(pd.NaT)
                                continue
                            
                            # Try various custom parsing approaches
                            parsed_date = None
                            
                            # Handle common separators
                            for sep in ['/', '-', '.', ' ']:
                                if sep in value_str:
                                    parts = value_str.split(sep)
                                    if len(parts) >= 3:
                                        try:
                                            # Try different orders
                                            for order in [(0, 1, 2), (1, 0, 2), (2, 1, 0), (2, 0, 1)]:
                                                day_idx, month_idx, year_idx = order
                                                day, month, year = int(parts[day_idx]), int(parts[month_idx]), int(parts[year_idx])
                                                
                                                # Fix 2-digit years
                                                if year < 100:
                                                    year = 2000 + year if year < 50 else 1900 + year
                                                
                                                if 1 <= day <= 31 and 1 <= month <= 12 and 1900 <= year <= 2100:
                                                    parsed_date = datetime(year, month, day)
                                                    break
                                            if parsed_date:
                                                break
                                        except:
                                            continue
                            
                            # If still not parsed, try pandas one more time with infer_datetime_format
                            if not parsed_date:
                                try:
                                    parsed_date = pd.to_datetime(value_str, infer_datetime_format=True)
                                except:
                                    pass
                            
                            custom_parsed.append(parsed_date if parsed_date else pd.NaT)
                        
                        custom_datetime = pd.Series(custom_parsed, index=original_col.index)
                        if custom_datetime.notna().sum() > 0:
                            if datetime_col is None or custom_datetime.notna().sum() > datetime_col.notna().sum():
                                datetime_col = custom_datetime
                    except Exception as e:
                        warnings.warn(f"Custom parsing failed for column '{col}': {str(e)}")
                
                # Apply the best datetime conversion found
                if datetime_col is not None and datetime_col.notna().sum() > 0:
                    # Format to dd/mm/yyyy
                    if target_format == 'dd/mm/yyyy':
                        # Only format non-null datetime values
                        mask = datetime_col.notna()
                        if mask.any():
                            df_copy.loc[mask, col] = datetime_col.loc[mask].dt.strftime('%d/%m/%Y')
                            
                            # Keep original values where conversion failed
                            failed_mask = ~mask & original_col.notna()
                            if failed_mask.any() and not force_conversion:
                                df_copy.loc[failed_mask, col] = original_col.loc[failed_mask]
                            
                            print(f"✅ Successfully converted {mask.sum()} values in column '{col}' to {target_format}")
                        else:
                            warnings.warn(f"No valid dates found in column '{col}' after conversion attempts")
                else:
                    if force_conversion:
                        warnings.warn(f"Failed to convert any values in column '{col}' to date format")
                    else:
                        print(f"⚠️ Could not convert column '{col}' - keeping original values")
                        
            except Exception as e:
                warnings.warn(f"Error converting column '{col}' to date: {str(e)}")
                continue
        
        return df_copy
    
    def remove_empty_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove completely empty/null/blank rows."""
        if df is None or df.empty:
            return df
        return df.dropna(how='all')

    def remove_empty_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove completely empty/null/blank columns."""
        if df is None or df.empty:
            return df
        return df.dropna(axis=1, how='all')
    
    def trim_rows(self, df: pd.DataFrame, top_rows: int = 0, bottom_rows: int = 0) -> pd.DataFrame:
        """
        Remove specified number of rows from top and/or bottom.
        
        Args:
            df: Input DataFrame
            top_rows: Number of rows to remove from top
            bottom_rows: Number of rows to remove from bottom
            
        Returns:
            DataFrame with rows trimmed
        """
        if df is None or df.empty:
            return df
            
        start_idx = top_rows
        end_idx = len(df) - bottom_rows if bottom_rows > 0 else len(df)
        
        if start_idx >= end_idx:
            raise ValueError("Cannot remove more rows than available in DataFrame")
        
        return df.iloc[start_idx:end_idx].reset_index(drop=True)
    
    def load_csv(self, file_path: str, top_rows: int = 0, bottom_rows: int = 0, **kwargs) -> pd.DataFrame:
        """
        Load CSV file with optional row trimming.
        
        Args:
            file_path: Path to CSV file
            top_rows: Number of rows to remove from top
            bottom_rows: Number of rows to remove from bottom
            **kwargs: Additional arguments for pd.read_csv
            
        Returns:
            DataFrame from CSV
        """
        try:
            df = pd.read_csv(file_path, **kwargs)
            
            if top_rows > 0 or bottom_rows > 0:
                df = self.trim_rows(df, top_rows, bottom_rows)
            
            return df
        except Exception as e:
            raise Exception(f"Error loading CSV file: {str(e)}")
    
    def load_excel(self, file_path: str, top_rows: int = 0, bottom_rows: int = 0, **kwargs) -> pd.DataFrame:
        """
        Load Excel file with optional row trimming.
        
        Args:
            file_path: Path to Excel file
            top_rows: Number of rows to remove from top
            bottom_rows: Number of rows to remove from bottom
            **kwargs: Additional arguments for pd.read_excel
            
        Returns:
            DataFrame from Excel
        """
        try:
            df = pd.read_excel(file_path, **kwargs)
            
            if top_rows > 0 or bottom_rows > 0:
                df = self.trim_rows(df, top_rows, bottom_rows)
            
            return df
        except Exception as e:
            raise Exception(f"Error loading Excel file: {str(e)}")
    
    def flatten_json(self, data: Union[Dict, List], parent_key: str = '', separator: str = '_') -> Dict:
        """
        Flatten nested JSON data into a flat dictionary.
        
        Args:
            data: JSON data (dict or list)
            parent_key: Parent key for nested flattening
            separator: Separator for nested keys
            
        Returns:
            Flattened dictionary
        """
        items = []
        
        if isinstance(data, dict):
            for key, value in data.items():
                new_key = f"{parent_key}{separator}{key}" if parent_key else key
                
                if isinstance(value, (dict, list)):
                    items.extend(self.flatten_json(value, new_key, separator).items())
                else:
                    items.append((new_key, value))
                    
        elif isinstance(data, list):
            for i, value in enumerate(data):
                new_key = f"{parent_key}{separator}{i}" if parent_key else str(i)
                
                if isinstance(value, (dict, list)):
                    items.extend(self.flatten_json(value, new_key, separator).items())
                else:
                    items.append((new_key, value))
        else:
            items.append((parent_key, data))
        
        return dict(items)
    
    def load_json(self, file_path: str, flatten: bool = False, extract_path: str = None) -> pd.DataFrame:
        """
        Load JSON file with optional flattening and path extraction.
        
        Args:
            file_path: Path to JSON file
            flatten: Whether to flatten nested JSON
            extract_path: JSON path to extract (e.g., 'data.items')
            
        Returns:
            DataFrame from JSON
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract specific path if specified
            if extract_path:
                path_parts = extract_path.split('.')
                for part in path_parts:
                    if isinstance(data, dict) and part in data:
                        data = data[part]
                    elif isinstance(data, list) and part.isdigit():
                        data = data[int(part)]
                    else:
                        raise ValueError(f"Path '{extract_path}' not found in JSON data")
            
            # Handle different JSON structures
            if isinstance(data, list):
                if flatten:
                    flattened_data = [self.flatten_json(item) for item in data]
                    df = pd.DataFrame(flattened_data)
                else:
                    df = pd.json_normalize(data)
            elif isinstance(data, dict):
                if flatten:
                    flattened_data = self.flatten_json(data)
                    df = pd.DataFrame([flattened_data])
                else:
                    df = pd.json_normalize(data)
            else:
                # Single value
                df = pd.DataFrame([{'value': data}])
            
            return df
        except Exception as e:
            raise Exception(f"Error loading JSON file: {str(e)}")
    
    def xml_to_dict_elementtree(self, element: ET.Element) -> Dict:
        """Convert XML element to dictionary using ElementTree."""
        result = {}
        
        # Add attributes
        if element.attrib:
            result.update({f"@{k}": v for k, v in element.attrib.items()})
        
        # Add text content
        if element.text and element.text.strip():
            if len(element) == 0:  # No children
                return element.text.strip()
            else:
                result['#text'] = element.text.strip()
        
        # Add children
        for child in element:
            child_data = self.xml_to_dict_elementtree(child)
            
            if child.tag in result:
                if not isinstance(result[child.tag], list):
                    result[child.tag] = [result[child.tag]]
                result[child.tag].append(child_data)
            else:
                result[child.tag] = child_data
        
        return result
    
    def xml_to_dict_dom(self, node) -> Dict:
        """Convert XML node to dictionary using DOM."""
        result = {}
        
        if node.nodeType == node.ELEMENT_NODE:
            # Add attributes
            if node.attributes:
                result.update({f"@{attr.name}": attr.value for attr in node.attributes.values()})
            
            # Add child elements
            for child in node.childNodes:
                if child.nodeType == child.TEXT_NODE:
                    text = child.nodeValue.strip()
                    if text:
                        result['#text'] = text
                elif child.nodeType == child.ELEMENT_NODE:
                    child_data = self.xml_to_dict_dom(child)
                    tag_name = child.nodeName
                    
                    if tag_name in result:
                        if not isinstance(result[tag_name], list):
                            result[tag_name] = [result[tag_name]]
                        result[tag_name].append(child_data)
                    else:
                        result[tag_name] = child_data
        
        return result
    
    class XMLSAXHandler(xml.sax.ContentHandler):
        """SAX handler for XML parsing."""
        
        def __init__(self):
            self.stack = []
            self.data = {}
            self.current_data = ""
            
        def startElement(self, name, attrs):
            element = {'tag': name, 'attrs': dict(attrs), 'children': [], 'text': ''}
            if self.stack:
                self.stack[-1]['children'].append(element)
            else:
                self.data = element
            self.stack.append(element)
            self.current_data = ""
            
        def endElement(self, name):
            if self.stack:
                element = self.stack.pop()
                element['text'] = self.current_data.strip()
            self.current_data = ""
            
        def characters(self, content):
            self.current_data += content
    
    def load_xml(self, file_path: str, parsing_method: str = 'elementtree', 
                 flatten: bool = False, xpath_query: str = None) -> pd.DataFrame:
        """
        Load XML file with various parsing methods.
        
        Args:
            file_path: Path to XML file
            parsing_method: XML parsing method ('elementtree', 'dom', 'sax', 'iterparse', 'xpath')
            flatten: Whether to flatten nested XML
            xpath_query: XPath query for targeted extraction
            
        Returns:
            DataFrame from XML
        """
        try:
            if parsing_method == 'elementtree':
                tree = ET.parse(file_path)
                root = tree.getroot()
                
                if xpath_query:
                    elements = root.findall(xpath_query)
                    data = [self.xml_to_dict_elementtree(elem) for elem in elements]
                else:
                    # For single root element, try to find repeating child elements
                    child_elements = list(root)
                    if child_elements and len(child_elements) > 1:
                        # Multiple child elements - treat each as a record
                        data = [self.xml_to_dict_elementtree(elem) for elem in child_elements]
                    else:
                        # Single element or no children - treat root as single record
                        data = self.xml_to_dict_elementtree(root)
                    
            elif parsing_method == 'dom':
                dom = xml.dom.minidom.parse(file_path)
                root = dom.documentElement
                
                # Check for multiple child elements
                element_children = [child for child in root.childNodes if child.nodeType == child.ELEMENT_NODE]
                if len(element_children) > 1:
                    data = [self.xml_to_dict_dom(elem) for elem in element_children]
                else:
                    data = self.xml_to_dict_dom(root)
                
            elif parsing_method == 'sax':
                handler = self.XMLSAXHandler()
                xml.sax.parse(file_path, handler)
                data = handler.data
                
            elif parsing_method == 'iterparse':
                data = []
                root_tag = None
                for event, elem in ET.iterparse(file_path, events=('start', 'end')):
                    if event == 'start' and root_tag is None:
                        root_tag = elem.tag
                    elif event == 'end' and elem.tag != root_tag:
                        # Process non-root elements
                        elem_data = self.xml_to_dict_elementtree(elem)
                        data.append({elem.tag: elem_data})
                        elem.clear()  # Free memory
                        
            elif parsing_method == 'xpath':
                try:
                    from lxml import etree
                    tree = etree.parse(file_path)
                    if xpath_query:
                        elements = tree.xpath(xpath_query)
                        data = [self._lxml_to_dict(elem) for elem in elements]
                    else:
                        raise ValueError("XPath query required for xpath parsing method")
                except ImportError:
                    raise ImportError("lxml package required for XPath parsing")
            else:
                raise ValueError(f"Unsupported parsing method: {parsing_method}")
            
            # Convert to DataFrame with better handling
            try:
                if isinstance(data, list) and len(data) > 0:
                    if flatten:
                        flattened_data = [self.flatten_json(item) for item in data]
                        df = pd.DataFrame(flattened_data)
                    else:
                        # Use json_normalize with max_level to avoid deep nesting
                        df = pd.json_normalize(data, max_level=2)
                        
                elif isinstance(data, dict):
                    if flatten:
                        flattened_data = self.flatten_json(data)
                        df = pd.DataFrame([flattened_data])
                    else:
                        # Normalize the dictionary
                        df = pd.json_normalize([data], max_level=2)
                else:
                    df = pd.DataFrame([{'value': str(data)}])
                    
                # Clean up column names - remove dots and make them more readable
                df.columns = [col.replace('.', '_') for col in df.columns]
                
            except Exception as e:
                # Fallback: convert everything to strings
                warnings.warn(f"Complex XML structure detected, converting to string representation: {str(e)}")
                if isinstance(data, list):
                    df = pd.DataFrame([{'xml_data': str(item)} for item in data])
                else:
                    df = pd.DataFrame([{'xml_data': str(data)}])
            
            return df
        except Exception as e:
            raise Exception(f"Error loading XML file: {str(e)}")
    
    def _lxml_to_dict(self, element) -> Dict:
        """Convert lxml element to dictionary (for XPath parsing)."""
        result = {}
        
        # Add attributes
        if element.attrib:
            result.update({f"@{k}": v for k, v in element.attrib.items()})
        
        # Add text content
        if element.text and element.text.strip():
            if len(element) == 0:  # No children
                return element.text.strip()
            else:
                result['#text'] = element.text.strip()
        
        # Add children
        for child in element:
            child_data = self._lxml_to_dict(child)
            tag_name = child.tag
            
            if tag_name in result:
                if not isinstance(result[tag_name], list):
                    result[tag_name] = [result[tag_name]]
                result[tag_name].append(child_data)
            else:
                result[tag_name] = child_data
        
        return result
    
    def load_from_api(self, url: str, headers: Dict = None, params: Dict = None, 
                     method: str = 'GET', data_format: str = 'json', 
                     extract_path: str = None, flatten: bool = False) -> pd.DataFrame:
        """
        Load data from API endpoint.
        
        Args:
            url: API endpoint URL
            headers: HTTP headers
            params: URL parameters
            method: HTTP method (GET, POST)
            data_format: Expected data format ('json', 'xml')
            extract_path: Path to extract from response
            flatten: Whether to flatten nested data
            
        Returns:
            DataFrame from API response
        """
        try:
            if method.upper() == 'GET':
                response = requests.get(url, headers=headers, params=params, timeout=30)
            elif method.upper() == 'POST':
                response = requests.post(url, headers=headers, json=params, timeout=30)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            response.raise_for_status()
            
            if data_format.lower() == 'json':
                data = response.json()
                
                # Extract specific path if specified
                if extract_path:
                    path_parts = extract_path.split('.')
                    for part in path_parts:
                        if isinstance(data, dict) and part in data:
                            data = data[part]
                        elif isinstance(data, list) and part.isdigit():
                            data = data[int(part)]
                        else:
                            raise ValueError(f"Path '{extract_path}' not found in API response")
                
                # Convert to DataFrame
                if isinstance(data, list):
                    if flatten:
                        flattened_data = [self.flatten_json(item) for item in data]
                        df = pd.DataFrame(flattened_data)
                    else:
                        df = pd.json_normalize(data)
                else:
                    df = pd.DataFrame([{'value': data}])
                    
            elif data_format.lower() == 'xml':
                # Parse XML from response text
                root = ET.fromstring(response.text)
                data = self.xml_to_dict_elementtree(root)
                
                if flatten:
                    flattened_data = self.flatten_json(data)
                    df = pd.DataFrame([flattened_data])
                else:
                    df = pd.json_normalize(data)
            else:
                raise ValueError(f"Unsupported data format: {data_format}")
            
            return df
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"API request failed: {str(e)}")
        except json.JSONDecodeError as e:
            raise Exception(f"Failed to parse JSON response: {str(e)}")
        except ET.ParseError as e:
            raise Exception(f"Failed to parse XML response: {str(e)}")
        except Exception as e:
            raise Exception(f"Error loading data from API: {str(e)}")
    
    def process_data(
        self,
        df: pd.DataFrame,
        date_columns: List[str] = None,
        date_format: str = 'dd/mm/yyyy',
        remove_empty_columns: bool = False,
        force_date_conversion: bool = True
    ) -> pd.DataFrame:
        """
        Process DataFrame with configurable options.
        
        Args:
            df: Input DataFrame
            date_columns: List of column names to convert to dates
            date_format: Target date format (currently only supports dd/mm/yyyy)
            remove_empty_columns: Whether to remove completely empty columns
            force_date_conversion: If True, try multiple parsing methods aggressively
            
        Returns:
            Processed DataFrame
        """
        if df is None or df.empty:
            return df

        processed_df = df.copy()

        # Always remove empty rows (as per requirement)
        processed_df = self.remove_empty_rows(processed_df)

        # Remove columns only if requested by user
        if remove_empty_columns:
            processed_df = self.remove_empty_columns(processed_df)

        # Convert date columns if specified
        if date_columns:
            processed_df = self.convert_date_columns(
                processed_df,
                date_columns,
                date_format,
                force_conversion=force_date_conversion
            )

        return processed_df

    def get_file_info(self, file_path: str) -> Dict:
        """
        Get information about a file.
        
        Args:
            file_path: Path to file
            
        Returns:
            Dictionary with file information
        """
        import os
        
        file_ext = os.path.splitext(file_path)[1].lower().lstrip('.')
        file_size = os.path.getsize(file_path)
        
        return {
            'file_path': file_path,
            'file_extension': file_ext,
            'file_size_bytes': file_size,
            'supported': file_ext in self.supported_file_types
        }