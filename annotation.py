import streamlit as st
import pandas as pd
import json
from typing import List, Dict, Any
import os

# Configure page settings
st.set_page_config(
    page_title="Document Query Annotation Tool",
    page_icon="📝",
    layout="wide"
)

def load_data(uploaded_file) -> pd.DataFrame:
    """
    Load data from uploaded CSV file
    
    Args:
        uploaded_file: Streamlit uploaded file object
    
    Returns:
        pandas.DataFrame: Loaded data
    """
    try:
        # Read the CSV file
        print(uploaded_file)
        df = pd.read_csv(uploaded_file, sep=',')  # Assuming tab-separated based on your data
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

def initialize_annotations(df: pd.DataFrame) -> Dict[int, List[str]]:
    """
    Initialize annotations dictionary for storing queries per document
    
    Args:
        df: DataFrame containing documents
    
    Returns:
        Dict mapping document indices to list of queries
    """
    # Check if annotations exist in session state
    if 'annotations' not in st.session_state:
        st.session_state.annotations = {i: [] for i in range(len(df))}
    return st.session_state.annotations

def save_annotations_to_file(annotations: Dict[int, List[str]], filename: str = "annotations.json"):
    """
    Save annotations to a JSON file
    
    Args:
        annotations: Dictionary of annotations
        filename: Name of file to save to
    """
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(annotations, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        st.error(f"Error saving annotations: {str(e)}")
        return False

def load_annotations_from_file(filename: str = "annotations.json") -> Dict[int, List[str]]:
    """
    Load annotations from a JSON file
    
    Args:
        filename: Name of file to load from
    
    Returns:
        Dictionary of loaded annotations or empty dict if file doesn't exist
    """
    try:
        if os.path.exists(filename):
            with open(filename, 'r', encoding='utf-8') as f:
                # Convert string keys back to integers
                loaded = json.load(f)
                return {int(k): v for k, v in loaded.items()}
        return {}
    except Exception as e:
        st.error(f"Error loading annotations: {str(e)}")
        return {}

def display_document_preview(text: str, max_chars: int = 500) -> str:
    """
    Create a preview of document text
    
    Args:
        text: Full document text
        max_chars: Maximum characters to show in preview
    
    Returns:
        Truncated text with ellipsis if needed
    """
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."

def main():
    """
    Main application function
    """
    st.title("📝 Document Query Annotation Tool")
    st.markdown("---")
    
    # Sidebar for file upload and controls
    with st.sidebar:
        st.header("📁 Data Management")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload your document CSV file",
            type=['csv', 'tsv'],
            help="Upload a CSV or TSV file containing your documents"
        )
        
        # Load/Save annotations
        st.subheader("💾 Annotations")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Load Saved", help="Load previously saved annotations"):
                loaded_annotations = load_annotations_from_file()
                if loaded_annotations:
                    st.session_state.annotations = loaded_annotations
                    st.success("Annotations loaded!")
        
        with col2:
            if st.button("Save Progress", help="Save current annotations to file"):
                if 'annotations' in st.session_state:
                    if save_annotations_to_file(st.session_state.annotations):
                        st.success("Saved!")
    
    # Main content area
    if uploaded_file is not None:
        # Load the data
        df = load_data(uploaded_file)
        
        if not df.empty:
            # Initialize annotations
            annotations = initialize_annotations(df)
            
            # Display data overview
            st.subheader("📊 Dataset Overview")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Documents", len(df))
            with col2:
                annotated_count = sum(1 for queries in annotations.values() if queries)
                st.metric("Annotated Documents", annotated_count)
            with col3:
                completion_rate = (annotated_count / len(df)) * 100 if len(df) > 0 else 0
                st.metric("Completion Rate", f"{completion_rate:.1f}%")
            
            st.markdown("---")
            
            # Document navigation
            st.subheader("🔍 Document Navigation")
            
            # Document selector
            doc_index = st.selectbox(
                "Select document to annotate:",
                range(len(df)),
                format_func=lambda x: f"Document {x + 1} ({len(annotations[x])} queries)",
                help="Choose which document to annotate"
            )
            
            # Navigation buttons
            col1, col2, col3 = st.columns([1, 1, 4])
            with col1:
                if st.button("← Previous") and doc_index > 0:
                    st.rerun()
            with col2:
                if st.button("Next →") and doc_index < len(df) - 1:
                    st.rerun()
            
            st.markdown("---")
            
            # Display current document
            current_doc = df.iloc[doc_index]
            
            st.subheader(f"📄 Document {doc_index + 1}")
            
            # Document metadata in expandable section
            with st.expander("📋 Document Metadata", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    if 'url' in current_doc and pd.notna(current_doc['url']):
                        st.write(f"**URL:** {current_doc['url']}")
                    if 'date' in current_doc and pd.notna(current_doc['date']):
                        st.write(f"**Date:** {current_doc['date']}")
                    if 'language' in current_doc and pd.notna(current_doc['language']):
                        st.write(f"**Language:** {current_doc['language']}")
                
                with col2:
                    if 'token_count' in current_doc and pd.notna(current_doc['token_count']):
                        st.write(f"**Token Count:** {current_doc['token_count']}")
                    if 'file_path' in current_doc and pd.notna(current_doc['file_path']):
                        st.write(f"**File Path:** {current_doc['file_path']}")
            
            # Document text display
            st.subheader("📖 Document Content")
            
            # Show preview or full text based on user preference
            show_full_text = st.checkbox("Show full text", value=False)
            
            if 'text' in current_doc:
                document_text = current_doc['text']
                if show_full_text:
                    st.text_area(
                        "Full Document Text:",
                        value=document_text,
                        height=300,
                        disabled=True,
                        help="Full document content"
                    )
                else:
                    preview_text = display_document_preview(document_text)
                    st.text_area(
                        "Document Preview:",
                        value=preview_text,
                        height=200,
                        disabled=True,
                        help="Preview of document content (check 'Show full text' for complete content)"
                    )
            else:
                st.warning("No 'text' column found in the dataset.")
            
            st.markdown("---")
            
            # Query annotation section
            st.subheader("🏷️ Query Annotations")
            
            # Display current queries for this document
            current_queries = annotations[doc_index]
            
            if current_queries:
                st.write("**Current Queries:**")
                for i, query in enumerate(current_queries):
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.write(f"{i + 1}. {query}")
                    with col2:
                        # Delete button for each query
                        if st.button("🗑️", key=f"delete_{doc_index}_{i}", help="Delete this query"):
                            annotations[doc_index].pop(i)
                            st.rerun()
            else:
                st.info("No queries added yet for this document.")
            
            # Add new query
            st.write("**Add New Query:**")
            new_query = st.text_input(
                "Enter a query for this document:",
                key=f"query_input_{doc_index}",
                placeholder="e.g., What is the main topic of this document?",
                help="Enter a question or query related to this document"
            )
            
            col1, col2 = st.columns([1, 3])
            with col1:
                if st.button("➕ Add Query", disabled=not new_query.strip()):
                    if new_query.strip():
                        annotations[doc_index].append(new_query.strip())
                        # Clear the input by rerunning (input will reset due to key change)
                        st.success("Query added!")
                        st.rerun()
            
            # Bulk operations
            st.markdown("---")
            st.subheader("⚡ Bulk Operations")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("Clear All Queries", help="Remove all queries for current document"):
                    annotations[doc_index] = []
                    st.success("All queries cleared for current document!")
                    st.rerun()
            
            with col2:
                # Copy queries from another document
                if len(df) > 1:
                    source_doc = st.selectbox(
                        "Copy from document:",
                        [i for i in range(len(df)) if i != doc_index],
                        format_func=lambda x: f"Document {x + 1}",
                        key="copy_source"
                    )
                    if st.button("Copy Queries"):
                        if annotations[source_doc]:
                            annotations[doc_index].extend(annotations[source_doc])
                            st.success(f"Copied {len(annotations[source_doc])} queries!")
                            st.rerun()
                        else:
                            st.warning("Source document has no queries to copy.")
            
            with col3:
                # Export current annotations
                if st.button("📤 Export Annotations"):
                    # Create a summary of annotations
                    export_data = []
                    for idx, queries in annotations.items():
                        if queries:  # Only include documents with queries
                            export_data.append({
                                'document_index': idx,
                                'document_preview': display_document_preview(df.iloc[idx]['text'], 100) if 'text' in df.columns else 'N/A',
                                'queries': queries,
                                'query_count': len(queries)
                            })
                    
                    if export_data:
                        export_df = pd.DataFrame(export_data)
                        csv_data = export_df.to_csv(index=False)
                        
                        st.download_button(
                            label="📥 Download Annotations CSV",
                            data=csv_data,
                            file_name="document_annotations.csv",
                            mime="text/csv",
                            help="Download annotations as CSV file"
                        )
                    else:
                        st.warning("No annotations to export.")
            
            # Progress tracking
            st.markdown("---")
            st.subheader("📈 Progress Tracking")
            
            # Create progress visualization
            progress_data = []
            for i in range(len(df)):
                query_count = len(annotations[i])
                progress_data.append({
                    'Document': f"Doc {i + 1}",
                    'Queries': query_count,
                    'Status': 'Annotated' if query_count > 0 else 'Pending'
                })
            
            progress_df = pd.DataFrame(progress_data)
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Annotation Progress:**")
                st.dataframe(progress_df, hide_index=True)
            
            with col2:
                # Summary statistics
                total_queries = sum(len(queries) for queries in annotations.values())
                avg_queries = total_queries / len(df) if len(df) > 0 else 0
                
                st.write("**Statistics:**")
                st.write(f"• Total queries across all documents: {total_queries}")
                st.write(f"• Average queries per document: {avg_queries:.1f}")
                st.write(f"• Documents with queries: {annotated_count}/{len(df)}")
        
        else:
            st.error("Failed to load data. Please check your file format.")
    
    else:
        # Welcome screen when no file is uploaded
        st.info("👆 Please upload a CSV file to start annotating documents.")
        
        st.subheader("📖 How to use this tool:")
        st.markdown("""
        1. **Upload your document dataset** using the sidebar file uploader
        2. **Navigate through documents** using the dropdown or navigation buttons
        3. **View document content** in the text area (toggle full text view if needed)
        4. **Add queries** by typing them in the input field and clicking "Add Query"
        5. **Manage queries** by deleting individual queries or clearing all
        6. **Save your progress** regularly using the "Save Progress" button
        7. **Export annotations** when finished using the "Export Annotations" button
        
        **Expected CSV Format:**
        - The CSV should have a 'text' column containing the document content
        - Additional metadata columns (url, date, language, etc.) are optional
        - Use tab-separated format if your data contains commas within text
        """)

if __name__ == "__main__":
    main()