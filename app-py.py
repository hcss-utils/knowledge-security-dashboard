import plotly.graph_objects as go
import pandas as pd
import numpy as np
import re
import os
from sqlalchemy import create_engine, text
import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State
import plotly.express as px
from urllib.parse import quote_plus

def create_db_connection():
    """
    Create database connection.
    """
    try:
        password = "GoNKJWp64NkMr9UdgCnT"
        encoded_password = quote_plus(password)
        engine = create_engine(
            f"postgresql+psycopg2://postgres:{encoded_password}@138.201.62.161:5434/knowledge_security"
        )
        return engine
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return None

def fetch_category_data(engine, selected_db=None):
    """
    Fetch hierarchical category data with optional database filter.
    """
    if selected_db == 'ALL':
        selected_db = None

    query = """
    SELECT
        t.category,
        t.subcategory,
        t.sub_subcategory,
        COUNT(*) AS count
    FROM taxonomy t
    JOIN document_section_chunk dsc ON t.chunk_id = dsc.id
    JOIN document_section ds ON dsc.document_section_id = ds.id
    JOIN uploaded_document ud ON ds.uploaded_document_id = ud.id
    WHERE (:db IS NULL OR ud.database = :db)
    GROUP BY t.category, t.subcategory, t.sub_subcategory;
    """
    try:
        return pd.read_sql(text(query), engine, params={'db': selected_db})
    except Exception as e:
        print(f"Error fetching category data: {e}")
        return None

def fetch_text_chunks(engine, level, value, selected_db=None):
    """
    Fetch all relevant text chunks for a specific category level
    with optional database filter. No limit here.
    """
    if selected_db == 'ALL':
        selected_db = None

    if level not in ['category', 'subcategory', 'sub_subcategory']:
        return pd.DataFrame()

    query = f"""
    SELECT
        t.category,
        t.subcategory,
        t.sub_subcategory,
        dsc.content AS chunk_text,
        t.chunk_level_reasoning AS reasoning,
        ud.document_id,
        ud.database,
        ds.heading_title,
        ud.date,
        ud.author
    FROM taxonomy t
    JOIN document_section_chunk dsc ON t.chunk_id = dsc.id
    JOIN document_section ds ON dsc.document_section_id = ds.id
    JOIN uploaded_document ud ON ds.uploaded_document_id = ud.id
    WHERE t.{level} = :value
      AND (:db IS NULL OR ud.database = :db)
    ORDER BY ud.date DESC;
    """
    try:
        return pd.read_sql(text(query), engine, params={'value': value, 'db': selected_db})
    except Exception as e:
        print(f"Error fetching text chunks: {e}")
        return pd.DataFrame()

def hex_to_rgba(hex_color, alpha):
    """Convert hex color to rgba with alpha."""
    hex_color = hex_color.lstrip('#')
    if hex_color.startswith('rgba'):
        return hex_color

    if not re.match(r"[0-9a-fA-F]{6}", hex_color):
        hex_color = '000000'

    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)

    return f'rgba({r},{g},{b},{alpha})'

def process_data_for_sunburst(df):
    """Process DataFrame for sunburst visualization."""
    outer_counts = df.groupby(['category', 'subcategory', 'sub_subcategory'])['count'].sum().reset_index()
    total_count = outer_counts['count'].sum()

    middle_counts = outer_counts.groupby(['category', 'subcategory'])['count'].sum().reset_index()
    middle_counts['percentage'] = (middle_counts['count'] / total_count * 100).round(2)

    inner_counts = outer_counts.groupby('category')['count'].sum().reset_index()
    inner_counts['percentage'] = (inner_counts['count'] / total_count * 100).round(2)

    return outer_counts, middle_counts, inner_counts, total_count

def create_color_mapping(inner_counts, middle_counts, outer_counts):
    """Create color mapping with alpha variations."""
    base_colors = [
        px.colors.convert_colors_to_same_type(px.colors.qualitative.Set3[i % len(px.colors.qualitative.Set3)])[0][0]
        if isinstance(px.colors.qualitative.Set3[i % len(px.colors.qualitative.Set3)], list)
        else px.colors.convert_colors_to_same_type([px.colors.qualitative.Set3[i % len(px.colors.qualitative.Set3)]])[0][0]
        for i in range(len(inner_counts))
    ]
    base_colors = [c.lstrip('rgb(').rstrip(')') for c in base_colors]
    base_colors = [c.split(',') for c in base_colors]
    base_colors = [f'#{int(c[0]):02x}{int(c[1]):02x}{int(c[2]):02x}' for c in base_colors]

    color_map = {}
    for idx, category in enumerate(inner_counts['category']):
        color_map[category] = base_colors[idx]

        sub_mask = middle_counts['category'] == category
        n_subs = sub_mask.sum()
        sub_alphas = np.linspace(0.7, 0.9, n_subs)
        np.random.seed(42)
        np.random.shuffle(sub_alphas)

        for sub_idx, (_, sub_row) in enumerate(middle_counts[sub_mask].iterrows()):
            alpha = min(max(sub_alphas[sub_idx], 0.0), 1.0)
            color_map[sub_row['subcategory']] = hex_to_rgba(base_colors[idx], alpha)

            subsub_mask = outer_counts['subcategory'] == sub_row['subcategory']
            n_subsubs = subsub_mask.sum()
            subsub_alphas = np.linspace(0.4, 0.6, n_subsubs)
            np.random.shuffle(subsub_alphas)

            for subsub_idx, (_, subsub_row) in enumerate(outer_counts[subsub_mask].iterrows()):
                alpha = min(max(subsub_alphas[subsub_idx], 0.0), 1.0)
                color_map[subsub_row['sub_subcategory']] = hex_to_rgba(base_colors[idx], alpha)

    return color_map

def create_dash_app(engine):
    """Create Dash application with interactive visualization and pagination for text chunks."""
    app = dash.Dash(__name__)

    # Page size for text chunks
    PAGE_SIZE = 10
    display_map = {
        'google_scholar': 'Google Scholar',
        'semantic_scholar': 'Semantic Scholar'
    }

    # Initial data fetch (no filter)
    df = fetch_category_data(engine)
    if df is None or df.empty:
        print("No initial data fetched. Using empty data.")
        df = pd.DataFrame(columns=['category', 'subcategory', 'sub_subcategory', 'count'])

    outer_counts, middle_counts, inner_counts, total_count = process_data_for_sunburst(df)
    color_map = create_color_mapping(inner_counts, middle_counts, outer_counts)

    # Initial Sunburst figure
    fig = go.Figure(go.Sunburst(
        ids=[*inner_counts['category'],
             *middle_counts['subcategory'],
             *outer_counts['sub_subcategory']],
        labels=[*inner_counts['category'],
                *middle_counts['subcategory'],
                *outer_counts['sub_subcategory']],
        parents=[''] * len(inner_counts) +
                list(middle_counts['category']) +
                list(outer_counts['subcategory']),
        values=[*inner_counts['count'],
                *middle_counts['count'],
                *outer_counts['count']],
        branchvalues='total',
        textinfo='label+value+percent root',
        marker=dict(
            colors=[color_map[label] for label in [*inner_counts['category'],
                                                   *middle_counts['subcategory'],
                                                   *outer_counts['sub_subcategory']]]
        ),
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percentRoot:.1%}<extra></extra>'
    ))
    fig.update_layout(
        title=f"Knowledge Security Taxonomy Distribution",
        width=800,
        height=800
    )

    app.layout = html.Div([
        html.Div(id='top'),

        # Filter Section
        html.Div([
            dcc.Dropdown(
                id='database-dropdown',
                options=[],
                placeholder='Select Database',
                value='ALL'
            ),
            html.Button('Apply Filter', id='apply-filter-button', n_clicks=0),
            html.Div(id='query-result-stats')
        ], style={'margin-bottom': '20px'}),

        # Sunburst Chart
        html.Div([
            dcc.Graph(
                id='sunburst-chart',
                figure=fig,
                style={'height': '700px'}
            )
        ], style={'margin-bottom': '40px'}),

        html.Div(id='selection-title-container',
                 style={'background': 'white', 'zIndex': 100, 'padding': '10px 0'}),
        html.Div(id='selection-stats', style={'margin-bottom': '20px'}),

        # Timeline chart
        dcc.Graph(id='timeline-chart', style={'margin-bottom': '10px'}),

        # Timeline caption (sticky)
        html.Div(id='timeline-caption', style={
            'position': 'sticky',
            'top': 0,
            'background': 'white',
            'zIndex': 200,
            'borderBottom': '2px solid #ccc',
            'padding': '10px'
        }),

        # Top pagination controls
        html.Div([
            html.Button('Previous Page', id='prev-page-button', n_clicks=0),
            html.Span("Page 1", id='page-indicator', style={'margin': '0 10px'}),
            html.Button('Next Page', id='next-page-button', n_clicks=0),
        ], id='top-pagination-controls',
           style={'margin-bottom': '20px', 'display': 'flex', 'justify-content': 'center', 'align-items': 'center'}),

        html.Div(id='text-chunks-container'),

        # Bottom pagination controls
        html.Div([
            html.Button('Previous Page', id='prev-page-button-bottom', n_clicks=0),
            html.Span("Page 1", id='page-indicator-bottom', style={'margin': '0 10px'}),
            html.Button('Next Page', id='next-page-button-bottom', n_clicks=0),
        ], style={'margin-top': '20px', 'display': 'flex', 'justify-content': 'center', 'align-items': 'center'}),

        dcc.Store(id='filtered-chunks-store'),
        dcc.Store(id='current-page-store', data=0),

        html.Div(id='dummy-scroll-div', style={'display': 'none'})
    ], style={'max-width': '800px', 'margin': 'auto'})

    @app.callback(
        Output('database-dropdown', 'options'),
        Input('apply-filter-button', 'n_clicks')
    )
    def update_database_options(n_clicks):
        query = """
        SELECT DISTINCT ud.database
        FROM uploaded_document ud;
        """
        try:
            databases_df = pd.read_sql(text(query), engine)

            options = [{'label': display_map.get(db, db), 'value': db} for db in databases_df['database'].unique()]
            options.insert(0, {'label': 'All Databases', 'value': 'ALL'})
            return options
        except Exception as e:
            print(f"Error fetching database options: {e}")
            return []

    @app.callback(
        Output('sunburst-chart', 'figure'),
        Input('apply-filter-button', 'n_clicks'),
        State('database-dropdown', 'value')
    )
    def update_sunburst(n_clicks, selected_db):
        if n_clicks > 0:
            df = fetch_category_data(engine, selected_db)
            if df is None or df.empty:
                return go.Figure().update_layout(title="No data available for the selected filters.")

            outer_counts, middle_counts, inner_counts, total_count = process_data_for_sunburst(df)
            color_map = create_color_mapping(inner_counts, middle_counts, outer_counts)

            db_display = "All Databases" if selected_db == "ALL" else display_map.get(selected_db, selected_db)

            fig = go.Figure(go.Sunburst(
                ids=[*inner_counts['category'],
                     *middle_counts['subcategory'],
                     *outer_counts['sub_subcategory']],
                labels=[*inner_counts['category'],
                        *middle_counts['subcategory'],
                        *outer_counts['sub_subcategory']],
                parents=[''] * len(inner_counts) +
                        list(middle_counts['category']) +
                        list(outer_counts['subcategory']),
                values=[*inner_counts['count'],
                        *middle_counts['count'],
                        *outer_counts['count']],
                branchvalues='total',
                textinfo='label+value+percent root',
                marker=dict(
                    colors=[color_map[label] for label in [*inner_counts['category'],
                                                           *middle_counts['subcategory'],
                                                           *outer_counts['sub_subcategory']]]
                ),
                hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percentRoot:.1%}<extra></extra>'
            ))
            fig.update_layout(
                title=f"Knowledge Security Taxonomy (Database: {db_display})",
                width=800,
                height=800
            )
            return fig
        else:
            return dash.no_update

    @app.callback(
        [Output('timeline-chart', 'figure'),
         Output('timeline-caption', 'children')],
        [Input('sunburst-chart', 'clickData'),
         State('database-dropdown', 'value')]
    )
    def update_timeline(clickData, selected_db):
        if not clickData:
            return go.Figure(), ""

        df_current = fetch_category_data(engine, selected_db)
        if df_current is None or df_current.empty:
            return go.Figure(), ""

        outer_counts, middle_counts, inner_counts, total_count = process_data_for_sunburst(df_current)

        selected = clickData['points'][0]['label']
        if selected in inner_counts['category'].values:
            level = 'category'
        elif selected in middle_counts['subcategory'].values:
            level = 'subcategory'
        else:
            level = 'sub_subcategory'

        timeline_df = fetch_text_chunks(engine, level, selected, selected_db)
        if timeline_df.empty:
            return go.Figure(), ""

        # For knowledge security, dates might be years rather than full dates
        try:
            timeline_df['year'] = pd.to_datetime(timeline_df['date']).dt.year
            timeline_counts = timeline_df.groupby('year').size().reset_index(name='count')
            
            fig = px.line(timeline_counts, x='year', y='count', title="")
            fig.update_xaxes(type='category')
        except:
            # If date processing fails, create an empty figure
            fig = go.Figure()
            fig.update_layout(title="Timeline data not available")

        db_display = "All Databases" if selected_db == "ALL" else display_map.get(selected_db, selected_db)

        caption_text = f"Publication Timeline for {level}: {selected}. Filter: Database = {db_display}"

        return fig, caption_text

    @app.callback(
        Output('filtered-chunks-store', 'data'),
        [Input('sunburst-chart', 'clickData'),
         State('database-dropdown', 'value')]
    )
    def store_filtered_chunks(clickData, selected_db):
        if not clickData:
            return []

        df_current = fetch_category_data(engine, selected_db)
        if df_current is None or df_current.empty:
            return []

        outer_counts, middle_counts, inner_counts, total_count = process_data_for_sunburst(df_current)
        selected = clickData['points'][0]['label']

        level = None
        if selected in inner_counts['category'].values:
            level = 'category'
        elif selected in middle_counts['subcategory'].values:
            level = 'subcategory'
        elif selected in df_current['sub_subcategory'].values:
            level = 'sub_subcategory'
        else:
            return []

        filtered_df = fetch_text_chunks(engine, level, selected, selected_db)
        if filtered_df is None or filtered_df.empty:
            return []

        return filtered_df.to_dict('records')

    @app.callback(
        [
            Output('selection-title-container', 'children'),
            Output('selection-stats', 'children'),
            Output('text-chunks-container', 'children'),
            Output('page-indicator', 'children'),
            Output('page-indicator-bottom', 'children'),
            Output('current-page-store', 'data')
        ],
        [
            Input('filtered-chunks-store', 'data'),
            Input('prev-page-button', 'n_clicks'),
            Input('next-page-button', 'n_clicks'),
            Input('prev-page-button-bottom', 'n_clicks'),
            Input('next-page-button-bottom', 'n_clicks')
        ],
        [
            State('sunburst-chart', 'clickData'),
            State('database-dropdown', 'value'),
            State('current-page-store', 'data')
        ]
    )
    def update_selection_and_pagination(
        filtered_data,
        prev_clicks_top,
        next_clicks_top,
        prev_clicks_bottom,
        next_clicks_bottom,
        clickData,
        selected_db,
        current_page
    ):
        PAGE_SIZE = 10

        if not clickData or not filtered_data:
            return (
                'Select a taxonomy category',
                'No category selected',
                [],
                "Page 1",
                "Page 1",
                0
            )

        df_current = fetch_category_data(engine, selected_db)
        if df_current is None or df_current.empty:
            return (
                'No data available',
                'No categories found for the selected filters',
                [],
                "Page 1",
                "Page 1",
                0
            )

        outer_counts, middle_counts, inner_counts, total_count = process_data_for_sunburst(df_current)
        selected = clickData['points'][0]['label']

        if selected in inner_counts['category'].values:
            level = 'category'
        elif selected in middle_counts['subcategory'].values:
            level = 'subcategory'
        elif selected in df_current['sub_subcategory'].values:
            level = 'sub_subcategory'
        else:
            return (
                f"Invalid selection: {selected}",
                "Unable to determine category level",
                [],
                "Page 1",
                "Page 1",
                0
            )

        full_df = pd.DataFrame(filtered_data)
        if full_df.empty:
            return (
                f"No data found for {level}: {selected}",
                f"No text chunks available for {selected}",
                [],
                "Page 1",
                "Page 1",
                0
            )

        ctx = dash.callback_context
        triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None

        # If the data changed (new selection), reset page
        if triggered_id == 'filtered-chunks-store':
            current_page = 0
        else:
            # Check which pagination button was clicked
            if triggered_id in ['prev-page-button', 'prev-page-button-bottom']:
                current_page = max(current_page - 1, 0)
            elif triggered_id in ['next-page-button', 'next-page-button-bottom']:
                max_page = (len(full_df) - 1) // PAGE_SIZE
                current_page = min(current_page + 1, max_page)

        start_index = current_page * PAGE_SIZE
        end_index = start_index + PAGE_SIZE
        display_df = full_df.iloc[start_index:end_index]

        stats = html.Div([
            html.P(f"Total chunks in selection: {len(full_df)}", style={'fontSize': '1.2em'}),
            html.P(f"Level: {level}", style={'fontSize': '1.2em'}),
            html.P(f"Unique documents: {full_df['document_id'].nunique()}", style={'fontSize': '1.2em'})
        ])

        chunk_rows = []
        for i, row in display_df.iterrows():
            try:
                metadata = html.P([
                    html.B("Document ID: "), f"{row.get('document_id', 'N/A')} | ",
                    html.B("Database: "), f"{row.get('database', 'N/A')} | ",
                    html.B("Heading: "), f"{row.get('heading_title', 'N/A')} | ",
                    html.B("Date: "), f"{row.get('date', 'N/A')} | ",
                    html.B("Author: "), f"{row.get('author', 'N/A')}"
                ], style={'margin-bottom': '5px'})

                chunk_text = html.Div([
                    html.P(row.get('chunk_text', 'No text available'))
                ], style={'padding': '10px', 'border': '1px solid lightgray',
                          'margin-bottom': '10px', 'width': '50%'})

                reasoning_text = row.get('reasoning', '')
                formatted_reasoning = []
                if reasoning_text:
                    reasoning_text = re.sub(r'# Reasoning', '', reasoning_text).strip()
                    sections = re.split(r'(\d+\..*?:)', reasoning_text)
                    sections = [s.strip() for s in sections if s.strip()]

                    for section in sections:
                        match = re.match(r'(\d+\.)(.*?):(.*)', section, re.DOTALL)
                        if match:
                            number, bold_content, rest_content = match.groups()
                            formatted_reasoning.append(html.P(f"{number} {bold_content}:"))
                            rest_content = rest_content.strip()
                            if rest_content:
                                bullet_items = [
                                    html.P("• " + item.strip())
                                    for item in re.split(r'(?<!\d\.)\s*-\s*', rest_content)
                                    if item.strip()
                                ]
                                formatted_reasoning.extend(bullet_items)
                        else:
                            formatted_reasoning.append(html.P(section))
                chunk_reasoning = html.Div(
                    formatted_reasoning or [html.P("No reasoning available")],
                    style={'padding': '10px', 'border': '1px solid lightgray',
                           'margin-bottom': '10px', 'width': '50%'}
                )

                chunk_row = html.Div([
                    html.Div([metadata]),
                    html.Div([chunk_text, chunk_reasoning], style={'display': 'flex', 'width': '100%'})
                ], style={'margin-bottom': '20px', 'border-bottom': '2px solid #ccc'})

                chunk_rows.append(chunk_row)

            except Exception as e:
                print(f"Error processing chunk {i}: {e}")
                continue

        total_pages = (len(full_df) - 1) // PAGE_SIZE + 1
        page_text = f"Page {current_page + 1} of {total_pages}"

        return (
            html.H3(f"Selected {level}: {selected}", style={'fontSize': '24px'}),
            stats,
            chunk_rows if chunk_rows else [html.P("No valid chunks to display")],
            page_text,  # top indicator
            page_text,  # bottom indicator
            current_page
        )

    # Clientside callback to scroll to top pagination controls on page change
    app.clientside_callback(
        """
        function(page) {
            const el = document.getElementById('top-pagination-controls');
            if (el) {
                el.scrollIntoView({behavior: 'smooth'});
            }
            return '';
        }
        """,
        Output('dummy-scroll-div', 'children'),
        Input('current-page-store', 'data')
    )

    return app

# Create the application
engine = create_db_connection()
app = create_dash_app(engine)
server = app.server  # This line is important for Heroku

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8050))
    app.run_server(debug=True, host='0.0.0.0', port=port)
