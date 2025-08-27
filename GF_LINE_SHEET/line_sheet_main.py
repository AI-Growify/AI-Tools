# streamlit_app.py
import os, time, json
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
import google.generativeai as genai
import numpy as np

# ── 1) Init GenAI Client ──────────────────────────────────────────────────
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")
def run():
# ── Helper Functions ──────────────────────────────────────────────────────
    def clean_value(value, is_numeric=False, default_numeric=0):
        """Clean values to avoid NaN in output"""
        if pd.isna(value) or value == 'nan' or value == 'NaN' or str(value).strip() == '':
            return default_numeric if is_numeric else ""
        
        if is_numeric:
            try:
                return float(str(value).strip())
            except (ValueError, TypeError):
                return default_numeric
        
        return str(value).strip()

    def parse_size_and_quantity(size_string):
        size_string = size_string.strip()
        
        # Handle special cases
        if size_string.lower() == 'custom':
            return 'Custom', 0
        
        if '-' in size_string and size_string.count('-') == 1:
            parts = size_string.split('-')
            size_part = parts[0].strip()
            try:
                qty_part = int(parts[1].strip())
                return size_part, qty_part
            except ValueError:
                # If second part is not a number, treat whole thing as size
                return size_string, 0
        else:
            # No dash or multiple dashes, treat as size only
            return size_string, 0


    def sort_sizes_with_quantities(sizes_list):
        standard_order = ['XXS', 'XS', 'S', 'M', 'L', 'XL', 'XXL', 'XXXL', '2XL', '3XL', '4XL', '5XL']
        
        # Split and clean sizes
        size_strings = [s.strip() for s in str(sizes_list).split(',') if s.strip()]
        
        # Parse each size string to extract size and quantity
        parsed_sizes = []
        size_quantity_map = {}
        
        for size_string in size_strings:
            actual_size, quantity = parse_size_and_quantity(size_string)
            parsed_sizes.append(actual_size)
            size_quantity_map[actual_size] = quantity
        
        # Remove duplicates while preserving order
        unique_sizes = []
        seen = set()
        for size in parsed_sizes:
            if size not in seen:
                unique_sizes.append(size)
                seen.add(size)
        
        # Separate standard, numeric, and custom sizes
        standard_sizes = []
        numeric_sizes = []
        custom_sizes = []
        
        for size in unique_sizes:
            # Check if size matches any standard size (case insensitive)
            matched = False
            for idx, std_size in enumerate(standard_order):
                if size.upper() == std_size.upper():
                    standard_sizes.append((idx, size))
                    matched = True
                    break
            
            if not matched:
                # Check if it's a pure numeric size
                import re
                if re.match(r'^\d+$', size):
                    # Pure numeric size like "5", "10", "12"
                    numeric_sizes.append((int(size), size))
                elif re.match(r'^X\d+$', size.upper()):
                    # Sizes like "X10", "X20"
                    numbers = re.findall(r'\d+', size)
                    if numbers:
                        numeric_sizes.append((int(numbers[0]), size))
                    else:
                        custom_sizes.append(size)
                else:
                    # Other custom sizes
                    custom_sizes.append(size)
        
        # Sort each category
        standard_sizes.sort(key=lambda x: x[0])
        numeric_sizes.sort(key=lambda x: x[0])  # Sort numerically
        custom_sizes.sort()
        
        # Combine: standard sizes first, then numeric sizes, then custom sizes
        sorted_sizes = ([size for _, size in standard_sizes] + 
                    [size for _, size in numeric_sizes] + 
                    custom_sizes)
        
        return sorted_sizes, size_quantity_map


    def sort_sizes(sizes_list):
        """Legacy function - now just returns the sorted sizes without quantities"""
        sorted_sizes, _ = sort_sizes_with_quantities(sizes_list)
        return sorted_sizes

    def process_variants_with_inventory(df_raw):
        """Process variants and extract inventory quantities from size strings"""
        df_exploded_list = []
        
        for _, row in df_raw.iterrows():
            # Parse sizes and extract quantities
            sorted_sizes, size_quantity_map = sort_sizes_with_quantities(clean_value(row.get("size", "")))
            colors = [c.strip() for c in str(clean_value(row.get("colour", ""))).split(",") if c.strip()]
            
            # If no sizes or colors, create default entries
            if not sorted_sizes:
                sorted_sizes = [""]
                size_quantity_map = {"": 0}
            if not colors:
                colors = [""]
            
            # Create all combinations
            for size in sorted_sizes:
                for color in colors:
                    new_row = row.copy()
                    new_row["sizes_list"] = size  # Just the size (S, M, L, XL)
                    new_row["colours_list"] = color
                    new_row["extracted_quantity"] = size_quantity_map.get(size, 0)  # Extracted quantity
                    df_exploded_list.append(new_row)
        
        return pd.DataFrame(df_exploded_list)

    # ── 2) Enhanced UI Setup ─────────────────────────────────────────────  

    # Custom CSS for better styling
    st.markdown("""
    <style>
        .main-header {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 10px;
            color: white;
            text-align: center;
            margin-bottom: 2rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .step-header {
            background: linear-gradient(45deg, #f093fb 0%, #f5576c 100%);
            padding: 1rem;
            border-radius: 8px;
            color: white;
            margin: 1rem 0;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }
        .variant-card {
            background: #f8f9fa;
            padding: 1rem;
            border-radius: 8px;
            border-left: 4px solid #28a745;
            margin: 0.5rem 0;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        }
        .stats-box {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 1rem;
            border-radius: 8px;
            text-align: center;
            margin: 0.5rem;
        }
        .ai-status {
            padding: 0.5rem;
            border-radius: 5px;
            margin: 0.5rem 0;
        }
        .ai-enabled { background-color: #d4edda; color: #155724; }
        .ai-disabled { background-color: #f8d7da; color: #721c24; }
    </style>
    """, unsafe_allow_html=True)

    # Main Header
    st.markdown("""
    <div class="main-header">
        <h1>Advanced Shopify CSV Builder</h1>
        <p>Transform your product data into Shopify-ready imports with AI-powered descriptions and smart inventory management</p>
    </div>
    """, unsafe_allow_html=True)

    # AI Status Check
    if not GEMINI_API_KEY:
        st.markdown('<div class="ai-status ai-disabled">⚠️ AI Features Disabled - Missing GEMINI_API_KEY</div>', unsafe_allow_html=True)
        ai_enabled = False
    else:
        model = genai.GenerativeModel('models/gemini-2.5-flash')
        ai_enabled = True

    # ── 3) Enhanced Sidebar Configuration ──────────────────────────────────
    with st.sidebar:
        st.markdown("## Configuration")
        
        # Processing mode selection
        mode = st.radio(
            "AI Processing Mode:",
            options=[
                "Default template (no AI)",
                "Simple mode (first sentence + tags)",
                "Full AI mode (custom description + tags)"
            ],
            index=0 if not ai_enabled else 0,
            disabled=not ai_enabled,
            help="Choose how you want to process product descriptions"
        )
        
        st.markdown("---")
        
        # Brand customization
        st.markdown("### Brand Settings")
        vendor_name = st.text_input("Vendor Name", value="YourBrandName", help="This will appear as the vendor in Shopify")
        
        # Inventory settings (simplified)
        st.markdown("### Inventory Settings")
        inventory_policy = st.selectbox(
            "Inventory Policy",
            options=["deny", "continue"],
            index=0,
            help="continue = Allow sales when out of stock, deny = Stop sales when out of stock"
        )
        
        default_qty = st.number_input(
            "Fallback Quantity", 
            min_value=0, 
            value=10, 
            step=1,
            help="Used when no quantity is found in size data"
        )
        
        bulk_qty_mode = st.checkbox(
            "Override with Bulk Quantity", 
            help="Ignore extracted quantities and set same quantity for all variants"
        )
        
        if bulk_qty_mode:
            bulk_qty = st.number_input("Bulk Quantity", min_value=0, value=default_qty, step=1)
        
        # Price settings (simplified)
        st.markdown("### Price Settings")
        default_compare_price = st.number_input(
            "Default Compare At Price", 
            min_value=0.0, 
            value=0.0, 
            step=0.01, 
            format="%.2f"
        )
        
        bulk_compare_price_mode = st.checkbox(
            "Enable Bulk Compare Price", 
            help="Set same compare price for all variants"
        )
        
        if bulk_compare_price_mode:
            bulk_compare_price = st.number_input(
                "Bulk Compare Price", 
                min_value=0.0, 
                value=default_compare_price, 
                step=0.01, 
                format="%.2f"
            )
        
        st.markdown("---")
        
        # Enhanced file format info
        with st.expander("Size Data Format", expanded=False):
            st.markdown("""
            **Size Format Examples:**
            - `S-4,M-8,L-12,XL-16` → Sizes: S,M,L,XL with Quantities: 4,8,12,16
            - `XS-0,S-5,M-10,L-15,XL-20` → Auto-extracts quantities
            - `Custom,Small,Medium,Large` → Uses fallback quantity
            
            **Other Required Columns:**
            - `title` - Product name
            - `description` - Product description  
            - `colour` - Colors (comma-separated)
            - `product code` - SKU base
            - `product category` - Category
            - `type` - Product type
            - `published` - Status (active/inactive)
            """)
        
        # Processing tips
        with st.expander("Processing Tips", expanded=False):
            st.markdown("""
            **How it works:**
            1. **Size Parsing**: `S-4` becomes Size=`S`, Quantity=`4`
            2. **Smart Sorting**: Sizes ordered as XS, S, M, L, XL, XXL, etc.
            3. **Manual Override**: You can adjust any extracted quantities
            4. **Bulk Options**: Override extracted quantities if needed
            
            **Best Practices:**
            - Use consistent size format: `SIZE-QUANTITY`
            - Check extracted quantities before processing
            - Use bulk mode only when you want same qty for all
            """)

    # ── 4) File Upload Section ──────────────────────────────────────────
    st.markdown('<div class="step-header"><h2>Step 1: Upload Your Product Data</h2></div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([200, 1, 1])

    with col1:
        uploaded_file = st.file_uploader(
            "Choose your CSV or Excel file",
            type=["csv", "xlsx"],
            help="Upload a file containing your product data"
        )

    with col2:
        if uploaded_file:
            file_size = len(uploaded_file.getvalue()) / 1024  # KB
            st.metric("File Size", f"{file_size:.1f} KB")

    with col3:
        if uploaded_file:
            file_type = "Excel" if uploaded_file.name.lower().endswith(".xlsx") else "CSV"
            st.metric("File Type", file_type)

    if not uploaded_file:
        st.info("Please upload a CSV or Excel file to get started")
        st.stop()

    # ── 5) Load & Preview Data ──────────────────────────────────────────
    try:
        df_raw = pd.read_excel(uploaded_file) if uploaded_file.name.lower().endswith(".xlsx") else pd.read_csv(uploaded_file)
        
        # Display success metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown('<div class="stats-box"><h3>{}</h3><p>Total Products</p></div>'.format(len(df_raw)), unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="stats-box"><h3>{}</h3><p>Columns</p></div>'.format(len(df_raw.columns)), unsafe_allow_html=True)
        with col3:
            total_variants = sum(len(str(row.get('size', '')).split(',')) * len(str(row.get('colour', '')).split(',')) for _, row in df_raw.iterrows())
            st.markdown('<div class="stats-box"><h3>{}</h3><p>Est. Variants</p></div>'.format(total_variants), unsafe_allow_html=True)
        with col4:
            active_products = sum(1 for _, row in df_raw.iterrows() if str(row.get('published', '')).lower() == 'active')
            st.markdown('<div class="stats-box"><h3>{}</h3><p>Active Products</p></div>'.format(active_products), unsafe_allow_html=True)
        
        # Data preview with tabs
        tab1, tab2 = st.tabs(["Data Preview", "Column Analysis"])
        
        with tab1:
            st.dataframe(df_raw.head(10), use_container_width=True)
        
        with tab2:
            col_analysis = pd.DataFrame({
                'Column': df_raw.columns,
                'Data Type': df_raw.dtypes,
                'Non-Null Count': df_raw.count(),
                'Null Count': df_raw.isnull().sum(),
                'Sample Value': [str(df_raw[col].iloc[0]) if len(df_raw) > 0 else 'N/A' for col in df_raw.columns]
            })
            st.dataframe(col_analysis, use_container_width=True)

    except Exception as e:
        st.error(f"Could not load file: {e}")
        st.stop()

    # ── 6) AI Helper Functions ────────────────────────────────────────────

    def refine_and_tag(text: str) -> tuple[str, str]:
        if not ai_enabled:
            return text, ""

        prompt = (
            "You are a top-tier Shopify copywriter.\n"
            "1) Rewrite this product description to be clear, engaging, and on-brand.\n"
            "2) Then output on the next line exactly five comma-separated tags.\n\n"
            f"Original description:\n\"\"\"\n{text}\n\"\"\"\n\n"
            "Respond with exactly two lines:\n"
            "- Line 1: your rewritten description\n"
            "- Line 2: tag1,tag2,tag3,tag4,tag5"
        )

        try:
            response = model.generate_content(prompt)
            parts = (response.text or "").strip().split("\n", 1)
            return parts[0].strip(), (parts[1].strip() if len(parts) > 1 else "")
        except Exception as e:
            st.warning(f"AI processing failed: {e}")
            return text, ""

    def tags_only(text: str) -> str:
        if not ai_enabled:
            return ""

        prompt = (
            "You are an expert Shopify copywriter.\n"
            "Suggest exactly five comma-separated Shopify tags for this product description:\n\n"
            f"\"\"\"\n{text}\n\"\"\"\n\n"
            "Respond with a single line:\n"
            "tag1,tag2,tag3,tag4,tag5"
        )

        try:
            response = model.generate_content(prompt)
            return (response.text or "").strip()
        except Exception as e:
            st.warning(f"AI tag generation failed: {e}")
            return ""

    # ── 7) Processing Section ──────────────────────────────────────────
    st.markdown('<div class="step-header"><h2>Step 2: Process Your Data</h2></div>', unsafe_allow_html=True)

    process_button = st.button("Start Processing", type="primary", use_container_width=True)
    if process_button or 'processed_data' in st.session_state:
        
        # Only do AI processing if button was clicked (not on rerun)
        if process_button:
            df = df_raw.copy()
            n = len(df)
            
            # Processing progress with detailed status
            progress_container = st.container()
            with progress_container:
                progress_bar = st.progress(0)
                status_text = st.empty()
                custom_descs, all_tags = [], []

                with st.spinner("AI is working its magic..."):
                    for i, (_, row) in enumerate(df.iterrows()):
                        title = row.get('title', 'Unknown')
                        if pd.isna(title) or title is None:
                            title = 'Unknown'
                        status_text.text(f"Processing product {i+1}/{n}: {str(title)[:30]}...")
                        
                        original = clean_value(row.get("description", ""))
                        if mode == "Default template (no AI)":
                            desc = f"{clean_value(row.get('title', ''))}"
                            tags = ""
                        elif mode == "Simple mode (first sentence + tags)":
                            first_sent = original.split(".", 1)[0].strip()
                            desc = first_sent
                            tags = tags_only(first_sent)
                        else:
                            desc, tags = refine_and_tag(original)

                        custom_descs.append(desc)
                        all_tags.append(tags)
                        progress_bar.progress((i + 1) / n)
                        time.sleep(0.1)  # Reduced sleep time

                status_text.text("processing complete!")

            df["custom_description"] = custom_descs
            df["ai_tags"] = all_tags

            # Process variants with extracted inventory quantities
            df = process_variants_with_inventory(df)
            
            # ── Corrected Handle Generation ──
            # Generate handles based on title + sku combination
            df["Handle"] = (df["title"].astype(str).fillna("").str.strip() + "-" + 
                            df.get("product code", pd.Series(dtype=str)).fillna("").astype(str).str.strip())

            # Clean handles for Shopify format
            df["Handle"] = (df["Handle"]
                            .str.replace(r"[^\w\s-]", "", regex=True)  # Remove special chars
                            .str.replace(r"\s+", "-", regex=True)      # Replace spaces with dashes
                            .str.lower()
                            .str.replace(r"-+", "-", regex=True)       # Remove multiple dashes
                            .str.strip("-"))                           # Remove leading/trailing dashes
            
            # Store processed data in session state
            st.session_state.processed_data = df
            st.session_state.variant_quantities = {}
            st.session_state.variant_compare_prices = {}
        else:
            # Use existing processed data
            df = st.session_state.processed_data

        # ── 8) Enhanced Inventory and Compare Price Management ──────────────────────────────
        st.markdown('<div class="step-header"><h2>Step 3: Manage Inventory & Pricing</h2></div>', unsafe_allow_html=True)
        
        # ── Updated unique variants logic with extracted quantities ──
        if 'unique_variants_processed' not in st.session_state:
            unique_variants = []
            variant_products = {}
            extracted_quantities = {}
            
            # Group by handle first, then get variants
            for handle, group in df.groupby("Handle"):
                title = clean_value(group.iloc[0]["title"])  # Get product title from first row
                
                for _, row in group.iterrows():
                    size = clean_value(row['sizes_list'])
                    color = clean_value(row['colours_list'])
                    extracted_qty = row.get('extracted_quantity', 0)
                    
                    variant_key = (size, color, title)
                    if variant_key not in unique_variants:
                        unique_variants.append(variant_key)
                        variant_products[variant_key] = title
                        extracted_quantities[variant_key] = extracted_qty
            
            st.session_state.unique_variants = unique_variants
            st.session_state.variant_products = variant_products
            st.session_state.extracted_quantities = extracted_quantities
            st.session_state.unique_variants_processed = True
        else:
            unique_variants = st.session_state.unique_variants
            variant_products = st.session_state.variant_products
            extracted_quantities = st.session_state.extracted_quantities

        # Initialize quantities and compare prices with extracted values
        if 'variant_quantities' not in st.session_state:
            st.session_state.variant_quantities = {}
            # HARD MAP: Always use extracted quantities first, fallback to default only if no extraction
            for size, color, title in unique_variants:
                variant_key = f"{size}|{color}|{title}"
                extracted_qty = extracted_quantities.get((size, color, title), 0)
                # HARD MAPPING: Use extracted quantity directly, or default if extraction is 0
                st.session_state.variant_quantities[variant_key] = extracted_qty if extracted_qty > 0 else default_qty

        if 'variant_compare_prices' not in st.session_state:
            st.session_state.variant_compare_prices = {}    
            for size, color, title in unique_variants:
                variant_key = f"{size}|{color}|{title}"
                st.session_state.variant_compare_prices[variant_key] = default_compare_price
                
        for size, color, title in unique_variants:
            variant_key = f"{size}|{color}|{title}"
            extracted_qty = extracted_quantities.get((size, color, title), 0)
            
            # FORCE MAPPING: If we have extracted quantity, always use it (unless user has manually changed it)
            if extracted_qty > 0:
                # Only update if current value is still the default (meaning user hasn't manually changed it)
                current_value = st.session_state.variant_quantities.get(variant_key, default_qty)
                if current_value == default_qty or variant_key not in st.session_state.variant_quantities:
                    st.session_state.variant_quantities[variant_key] = extracted_qty

        # Show extraction summary
        if any(extracted_quantities.values()):
            st.info("Inventory quantities have been extracted from your size data and pre-populated. You can adjust them below.")
            
            # Show extracted quantities summary
            with st.expander("Extracted Inventory Summary", expanded=False):
                summary_data = []
                for (size, color, title), qty in extracted_quantities.items():
                    if qty > 0:  # Only show sizes that had quantities
                        summary_data.append({
                            'Product': title,
                            'Size': size,
                            'Color': color if color else 'N/A',
                            'Extracted Quantity': qty
                        })
                
                if summary_data:
                    summary_df = pd.DataFrame(summary_data)
                    st.dataframe(summary_df, use_container_width=True)

        # Bulk mode handling (simplified)
        if bulk_qty_mode:
            st.info(f"Bulk mode enabled: Setting {bulk_qty} for all variants")
            for size, color, title in unique_variants:
                variant_key = f"{size}|{color}|{title}"
                st.session_state.variant_quantities[variant_key] = bulk_qty

        if bulk_compare_price_mode:
            st.info(f"Bulk compare price mode enabled: Setting ${bulk_compare_price:.2f} for all variants")
            for size, color, title in unique_variants:
                variant_key = f"{size}|{color}|{title}"
                st.session_state.variant_compare_prices[variant_key] = bulk_compare_price

        # Manual editing interface (enhanced)
        if not bulk_qty_mode or not bulk_compare_price_mode:
            # Group variants by product for better organization
            if 'products_variants_grouped' not in st.session_state:
                products_variants = {}
                for size, color, title in unique_variants:
                    if title not in products_variants:
                        products_variants[title] = []
                    products_variants[title].append((size, color))
                st.session_state.products_variants_grouped = products_variants
            else:
                products_variants = st.session_state.products_variants_grouped
            
            st.markdown("### Individual Variant Management")
            st.markdown("*Adjust quantities and compare prices for each variant below*")
            
            # Create expandable sections for each product
            for product_title, variants in products_variants.items():
                # Calculate summary stats for this product
                total_qty = 0
                total_variants = len(variants)
                
                for size, color in variants:
                    variant_key = f"{size}|{color}|{product_title}"
                    total_qty += st.session_state.variant_quantities.get(variant_key, 0)
                
                with st.expander(f"{product_title} ({total_variants} variants, {total_qty} total qty)", expanded=len(products_variants) <= 3):
                    
                    # Use form for batch updates per product
                    with st.form(key=f"form_{hash(product_title) % 10000}"):
                        
                        variant_inputs = {}
                        for idx, (size, color) in enumerate(variants):
                            variant_key = f"{size}|{color}|{product_title}"
                            current_qty = st.session_state.variant_quantities.get(variant_key, default_qty)
                            current_compare_price = st.session_state.variant_compare_prices.get(variant_key, default_compare_price)
                            
                            # Show extracted quantity if available
                            extracted_qty = extracted_quantities.get((size, color, product_title), 0)
                            col1, col2, col3 = st.columns([2, 2, 2])
                            
                            with col1:
                                st.markdown(f"**{size} / {color if color else 'N/A'}**")
                                if extracted_qty > 0:
                                    st.caption(f"↗Extracted: {extracted_qty}")
                            
                            with col2:
                                
                                current_qty = st.session_state.variant_quantities.get(variant_key, default_qty)
                                qty = st.number_input(
                                    f"Quantity",
                                    min_value=0, 
                                    value=int(current_qty),  # Use session state value directly
                                    step=1, 
                                    key=f"form_qty_{variant_key}_{hash(variant_key) % 10000}",
                                    help=f"Extracted from size data: {extracted_qty}" if extracted_qty > 0 else "Manual quantity"
                                )
                            
                            with col3:
                                compare_price = st.number_input(
                                    f"Compare Price ($)",
                                    min_value=0.0,
                                    value=float(current_compare_price),
                                    step=0.01,
                                    format="%.2f",
                                    key=f"form_price_{variant_key}_{hash(variant_key) % 10000}"
                                )
                            
                            variant_inputs[variant_key] = (qty, compare_price)
                        
                        # Submit button for this product
                        if st.form_submit_button(f"Update {product_title}", use_container_width=True):
                            # Batch update all variants for this product
                            for variant_key, (qty, compare_price) in variant_inputs.items():
                                st.session_state.variant_quantities[variant_key] = qty
                                st.session_state.variant_compare_prices[variant_key] = compare_price
                            st.success(f"Updated quantities and prices for {product_title}")
            
            # Enhanced table-based input for power users
            with st.expander("Power User: Spreadsheet-Style Editing", expanded=False):
                st.markdown("*Excel-like interface for bulk editing*")
                
                # Create a dataframe for editing
                variant_data = []
                for size, color, title in unique_variants:
                    variant_key = f"{size}|{color}|{title}"
                    extracted_qty = extracted_quantities.get((size, color, title), 0)
                    
                    # Use current session state value (which should now contain extracted quantities)
                    current_qty = st.session_state.variant_quantities.get(variant_key, default_qty)
                    
                    variant_data.append({
                        'Product': title,
                        'Size': size if size else 'N/A',
                        'Color': color if color else 'N/A',
                        'Extracted Qty': extracted_qty,
                        'Current Qty': current_qty,  # This will now show extracted qty
                        'Compare Price': st.session_state.variant_compare_prices.get(variant_key, default_compare_price),
                        'Key': variant_key
                    })
                
                variant_df = pd.DataFrame(variant_data)
                
                # Use data editor for super fast editing
                edited_variants = st.data_editor(
                    variant_df[['Product', 'Size', 'Color', 'Extracted Qty', 'Current Qty', 'Compare Price']], 
                    hide_index=True,
                    use_container_width=True,
                    key="variant_editor",
                    column_config={
                        'Extracted Qty': st.column_config.NumberColumn('Extracted Qty', disabled=True, help="Quantity extracted from size data"),
                        'Current Qty': st.column_config.NumberColumn('Current Qty', min_value=0, step=1),
                        'Compare Price': st.column_config.NumberColumn('Compare Price ($)', min_value=0.0, step=0.01, format="%.2f")
                    }
                )
                
                if st.button("Apply All Spreadsheet Changes", key="apply_table", type="primary"):
                    for i, variant_key in enumerate(variant_df['Key']):
                        st.session_state.variant_quantities[variant_key] = int(edited_variants.iloc[i]['Current Qty'])
                        st.session_state.variant_compare_prices[variant_key] = float(edited_variants.iloc[i]['Compare Price'])
                    st.success("Applied all spreadsheet changes!")
                    st.rerun()

        # Apply quantities and compare prices to dataframe - FIXED VERSION
        variant_qty_map = st.session_state.variant_quantities
        variant_compare_price_map = st.session_state.variant_compare_prices
        
        # Create variant key for mapping
        df["_variant_key"] = (df["sizes_list"].astype(str).fillna("").str.strip() + "|" + 
                            df["colours_list"].astype(str).fillna("").str.strip() + "|" + 
                            df["title"].astype(str).fillna("").str.strip())
        
        # CRITICAL FIX: Apply quantities and compare prices from session state
        def get_quantity(variant_key):
            return variant_qty_map.get(variant_key, default_qty)
        
        def get_compare_price(variant_key):
            return variant_compare_price_map.get(variant_key, default_compare_price)
        
        # Apply the mappings
        df["Variant Inventory Qty"] = df["_variant_key"].apply(get_quantity)
        df["Variant Compare At Price"] = df["_variant_key"].apply(get_compare_price)
        
        # Ensure data types are correct
        df["Variant Inventory Qty"] = pd.to_numeric(df["Variant Inventory Qty"], errors='coerce').fillna(0).astype(int)
        df["Variant Compare At Price"] = pd.to_numeric(df["Variant Compare At Price"], errors='coerce').fillna(0.0).astype(float)

        # ── Enhanced Final Dataset Generation with Proper Sorting and Inventory ──
        grouped_data = []

        for handle, group in df.groupby("Handle"):
            # CRITICAL FIX 1: Sort variants properly by size order, then by color
            # First, let's get the size order from the original data
            sizes_in_group = group['sizes_list'].unique()
            
            # Use our existing sort function to get proper order
            sorted_sizes_for_group = []
            if len(sizes_in_group) > 0:
                # Convert to comma-separated string and sort
                sizes_string = ','.join(sizes_in_group)
                sorted_sizes_for_group = sort_sizes(sizes_string)
            
            # Create a size order mapping
            size_order_map = {size: idx for idx, size in enumerate(sorted_sizes_for_group)}
            
            # Sort the group by size order, then by color
            def sort_key(row):
                size = row['sizes_list']
                color = row['colours_list']
                size_idx = size_order_map.get(size, 999)  # Unknown sizes go to end
                return (size_idx, color)
            
            # Apply sorting to the group
            group_list = list(group.iterrows())
            group_list.sort(key=lambda x: sort_key(x[1]))
            
            # Get product-level information from first row
            first_row = group_list[0][1]  # Get the row data from (index, row) tuple
            
            # CRITICAL FIX 2: Apply inventory quantities from session state before creating variants
            for idx, (_, row) in enumerate(group_list):
                variant_key = f"{row['sizes_list']}|{row['colours_list']}|{row['title']}"
                if variant_key in st.session_state.variant_quantities:
                    # Update the row's inventory quantity
                    group_list[idx] = (_, row.copy())
                    group_list[idx][1]['Variant Inventory Qty'] = st.session_state.variant_quantities[variant_key]
                if variant_key in st.session_state.variant_compare_prices:
                    # Update the row's compare price
                    if group_list[idx][1] is row:  # If we haven't copied yet
                        group_list[idx] = (_, row.copy())
                    group_list[idx][1]['Variant Compare At Price'] = st.session_state.variant_compare_prices[variant_key]
            
            # Create first row with complete product information
            first_row = group_list[0][1]  # Updated first row
            first_variant = {
                "Handle": clean_value(handle),
                "Title": clean_value(first_row["title"]),
                "Body (HTML)": f"<p>{clean_value(first_row['custom_description'])}</p>" if clean_value(first_row['custom_description']) else "",
                "Vendor": clean_value(vendor_name),
                "Product Category": clean_value(first_row.get("product category", "")),
                "Type": clean_value(first_row.get("type", "")),
                "Tags": clean_value(first_row["ai_tags"]),
                "Published": "TRUE" if str(clean_value(first_row.get("published", ""))).lower() == "active" else "FALSE",
                "Option1 Name": "Size",
                "Option1 Value": clean_value(first_row["sizes_list"]),
                "Option2 Name": "Color",
                "Option2 Value": clean_value(first_row["colours_list"]),
                "Option3 Name": "",
                "Option3 Value": "",
                "Variant SKU": f"{clean_value(first_row.get('product code', ''))}-{clean_value(first_row['sizes_list'])}",
                "Variant Grams": clean_value(0, is_numeric=True),
                "Variant Inventory Tracker": clean_value(first_row.get("Variant Inventory Tracker", "")),
                "Variant Inventory Qty": clean_value(first_row["Variant Inventory Qty"], is_numeric=True),
                "Variant Inventory Policy": inventory_policy,
                "Variant Fulfillment Service": "manual",
                "Variant Price": clean_value(first_row.get("Variant Price", 0), is_numeric=True),
                "Variant Compare At Price": clean_value(first_row["Variant Compare At Price"], is_numeric=True),
                "Variant Requires Shipping": "TRUE",
                "Variant Taxable": "TRUE",
                "Image Src": clean_value(first_row.get("Image Src", "")),
                "Image Position": clean_value(first_row.get("Image Position", "")),
                "Image Alt Text": clean_value(first_row.get("Image Alt Text", "")),
                "Gift Card": "FALSE",
                "SEO Title": clean_value(first_row.get("SEO Title", "")),
                "SEO Description": clean_value(first_row.get("SEO Description", "")),
                "Google Shopping / Google Product Category": clean_value(first_row.get("Google Shopping / Google Product Category", "")),
                "Google Shopping / Gender": clean_value(first_row.get("Google Shopping / Gender", "")),
                "Google Shopping / Age Group": clean_value(first_row.get("Google Shopping / Age Group", "")),
                "Google Shopping / MPN": clean_value(first_row.get("Google Shopping / MPN", "")),
                "Google Shopping / AdWords Grouping": clean_value(first_row.get("Google Shopping / AdWords Grouping", "")),
                "Google Shopping / AdWords Labels": clean_value(first_row.get("Google Shopping / AdWords Labels", "")),
                "Google Shopping / Condition": clean_value(first_row.get("Google Shopping / Condition", "")),
                "Google Shopping / Custom Product": clean_value(first_row.get("Google Shopping / Custom Product", "")),
                "Google Shopping / Custom Label 0": clean_value(first_row.get("Google Shopping / Custom Label 0", "")),
                "Google Shopping / Custom Label 1": clean_value(first_row.get("Google Shopping / Custom Label 1", "")),
                "Google Shopping / Custom Label 2": clean_value(first_row.get("Google Shopping / Custom Label 2", "")),
                "Google Shopping / Custom Label 3": clean_value(first_row.get("Google Shopping / Custom Label 3", "")),
                "Google Shopping / Custom Label 4": clean_value(first_row.get("Google Shopping / Custom Label 4", "")),
                "Variant Image": clean_value(first_row.get("Variant Image", "")),
                "Variant Weight Unit": clean_value(first_row.get("Variant Weight Unit", "")),
                "Variant Tax Code": clean_value(first_row.get("Variant Tax Code", "")),
                "Cost per item": clean_value(first_row.get("Cost per item", 0), is_numeric=True),
                "Status": clean_value(first_row.get("Status", "active"))
            }
            
            grouped_data.append(first_variant)
            
            # Create subsequent rows for remaining variants (if any)
            for _, row in group_list[1:]:  # Skip first row, process rest
                variant_row = {
                    "Handle": clean_value(handle),
                    "Title": "",  # Empty for additional variants
                    "Body (HTML)": "",
                    "Vendor": "",
                    "Product Category": "",
                    "Type": "",
                    "Tags": "",
                    "Published": "",
                    "Option1 Name": "",
                    "Option1 Value": clean_value(row["sizes_list"]),
                    "Option2 Name": "",
                    "Option2 Value": clean_value(row["colours_list"]),
                    "Option3 Name": "",
                    "Option3 Value": "",
                    "Variant SKU": f"{clean_value(row.get('product code', ''))}-{clean_value(row['sizes_list'])}",
                    "Variant Grams": clean_value(0, is_numeric=True),
                    "Variant Inventory Tracker": "",
                    "Variant Inventory Qty": clean_value(row["Variant Inventory Qty"], is_numeric=True),
                    "Variant Inventory Policy": inventory_policy,
                    "Variant Fulfillment Service": "manual",
                    "Variant Price": clean_value(row.get("Variant Price", 0), is_numeric=True),
                    "Variant Compare At Price": clean_value(row["Variant Compare At Price"], is_numeric=True),
                    "Variant Requires Shipping": "TRUE",
                    "Variant Taxable": "TRUE",
                    "Image Src": "",
                    "Image Position": "",
                    "Image Alt Text": "",
                    "Gift Card": "",
                    "SEO Title": "",
                    "SEO Description": "",
                    "Google Shopping / Google Product Category": "",
                    "Google Shopping / Gender": "",
                    "Google Shopping / Age Group": "",
                    "Google Shopping / MPN": "",
                    "Google Shopping / AdWords Grouping": "",
                    "Google Shopping / AdWords Labels": "",
                    "Google Shopping / Condition": "",
                    "Google Shopping / Custom Product": "",
                    "Google Shopping / Custom Label 0": "",
                    "Google Shopping / Custom Label 1": "",
                    "Google Shopping / Custom Label 2": "",
                    "Google Shopping / Custom Label 3": "",
                    "Google Shopping / Custom Label 4": "",
                    "Variant Image": "",
                    "Variant Weight Unit": "",
                    "Variant Tax Code": "",
                    "Cost per item": clean_value(0, is_numeric=True),
                    "Status": ""
                }
                grouped_data.append(variant_row)

        # Create the final output dataframe
        out = pd.DataFrame(grouped_data)
        
        # Final cleanup to ensure no NaN values in output
        for col in out.columns:
            if out[col].dtype == 'object':  # String columns
                out[col] = out[col].fillna('').astype(str)
                out[col] = out[col].replace(['nan', 'NaN', 'None'], '')
            else:  # Numeric columns
                out[col] = out[col].fillna(0)

        # ── 9) Results Display ──────────────────────────────────────────
        st.markdown('<div class="step-header"><h2>Step 4: Review & Download</h2></div>', unsafe_allow_html=True)
        
        # Final statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown('<div class="stats-box"><h3>{}</h3><p>Final Variants</p></div>'.format(len(out)), unsafe_allow_html=True)
        with col2:
            total_inventory = int(out["Variant Inventory Qty"].sum()) if len(out) > 0 else 0
            st.markdown('<div class="stats-box"><h3>{}</h3><p>Total Inventory</p></div>'.format(total_inventory), unsafe_allow_html=True)
        with col3:
            unique_products = out["Handle"].nunique() if len(out) > 0 else 0
            st.markdown('<div class="stats-box"><h3>{}</h3><p>Unique Products</p></div>'.format(unique_products), unsafe_allow_html=True)
        with col4:
            avg_price = out["Variant Price"].replace(0, pd.NA).mean()
            st.markdown('<div class="stats-box"><h3>{}</h3><p>Avg Price</p></div>'.format(f"${avg_price:.0f}" if pd.notna(avg_price) else "N/A"), unsafe_allow_html=True)

        # Tabbed results view
        tab1, tab2, tab3, tab4 = st.tabs(["Final Preview", "Inventory Summary", "AI Tags Overview", "Price Summary"])
        
        with tab1:
            st.dataframe(out.head(20), use_container_width=True)
        
        with tab2:
            try:
                inventory_summary = out.groupby(["Handle", "Title"]).agg({
                    "Variant Inventory Qty": ["sum", "count"],
                    "Variant Price": "first",
                    "Variant Compare At Price": "first"
                }).round(2)
                inventory_summary.columns = ["Total Qty", "Variants", "Price", "Compare Price"]
                st.dataframe(inventory_summary, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating inventory summary: {e}")
                st.dataframe(out[["Handle", "Title", "Variant Inventory Qty", "Variant Price", "Variant Compare At Price"]], use_container_width=True)
        
        with tab3:
            if ai_enabled and mode != "Default template (no AI)":
                tags_df = out[out["Tags"] != ""][["Title", "Tags"]].drop_duplicates()
                if len(tags_df) > 0:
                    st.dataframe(tags_df, use_container_width=True)
                else:
                    st.info("No AI tags generated")
            else:
                st.info("No AI tags generated in current mode")
        
        with tab4:
            try:
                price_summary = out[out["Variant Price"] > 0].groupby(["Handle", "Title"]).agg({
                    "Variant Price": ["min", "max", "mean"],
                    "Variant Compare At Price": ["min", "max", "mean"]
                }).round(2)
                price_summary.columns = ["Min Price", "Max Price", "Avg Price", "Min Compare", "Max Compare", "Avg Compare"]
                st.dataframe(price_summary, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating price summary: {e}")
                st.dataframe(out[["Handle", "Title", "Variant Price", "Variant Compare At Price"]], use_container_width=True)

        # Download section
        csv_data = out.to_csv(index=False).encode("utf-8")
        
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="Download Shopify CSV",
                data=csv_data,
                file_name=f"shopify_import_{time.strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                type="primary",
                use_container_width=True
            )
        
        with col2:
            if st.button("Process Another File", use_container_width=True):
                # Clear session state
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()

        # Success message
        st.success("Your Shopify CSV is ready! The file contains all variants with extracted inventory quantities and compare prices.")
        
        # Usage tips
        with st.expander("Next Steps & Tips"):
            st.markdown("""
            ### What to do next:
            1. **Download** your CSV file using the button above
            2. **Review** the data in Excel/Google Sheets if needed
            3. **Import** to Shopify via: Products → Import
            4. **Check** that all variants imported correctly
            
            ### Important Notes:
            - Make sure your Shopify store accepts the product categories used
            - Verify that all image URLs (if any) are accessible
            - Double-check pricing and inventory levels
            - Test with a small batch first if you have many products
            - Sizes are automatically sorted: XS, S, M, L, XL, XXL, XXXL, then custom sizes
            
            ### Enhanced Features:
            - **Smart Size Parsing**: Extracts quantities from 'S-4' format → Size: S, Qty: 4
            - **No NaN Values**: All empty fields are properly handled as blank entries
            - **Sorted Sizes**: Sizes appear in logical order (S, M, L, XL, etc.)
            - **Individual Compare Prices**: Set compare-at prices for each variant individually
            - **Extraction Summary**: Shows what quantities were automatically detected
            - **Spreadsheet Editing**: Power users can edit in table format for bulk changes
            """)