import pandas as pd
import plotly.express as px
import os


DATA_FOLDER = "data"
VIS_FOLDER = "vis"
SETS = ['SetA', 'SetB', 'SetC', 'SetD']

os.makedirs(VIS_FOLDER, exist_ok=True)

def save_responsive_html(fig, filename):
    fig.update_layout(autosize=True, margin=dict(l=20, r=20, t=40, b=20))
    fig.write_html(
        filename, 
        include_plotlyjs='cdn', 
        full_html=False, 
        config={'responsive': True}
    )

def load_and_clean(set_name):
    print(f"Feldolgozás: {set_name}...")
    e_path = os.path.join(DATA_FOLDER, f"SCLDDOS2024_{set_name}_events.csv")
    c_path = os.path.join(DATA_FOLDER, f"SCLDDoS2024_{set_name}_components.csv")
    
    if not os.path.exists(e_path) or not os.path.exists(c_path):
        return None
    
   
    e_df = pd.read_csv(e_path, usecols=['Attack ID', 'Start time', 'End time', 'Type', 'Attack code'])
    c_df = pd.read_csv(c_path, usecols=['Attack ID', 'Packet speed', 'Data speed', 'Avg packet len', 'Source IP count'])
    
    df = e_df.merge(c_df, on='Attack ID', how='left')
    df['Set'] = set_name
    
   
    df['Start time'] = pd.to_datetime(df['Start time'], errors='coerce')
    df['End time'] = pd.to_datetime(df['End time'], errors='coerce')
    df['Duration_sec'] = (df['End time'] - df['Start time']).dt.total_seconds()
    

    df['Type'] = df['Type'].astype('category')
    df['Attack code'] = df['Attack code'].astype('category')
    
    return df

def generate_suite(df, prefix):

    fig1 = px.pie(df, names='Type', title=f"Forgalmi típusok aránya - {prefix}")
    save_responsive_html(fig1, f"{VIS_FOLDER}/{prefix}_class_balance.html")

   
    fig2 = px.histogram(df.drop_duplicates('Attack ID'), x="Duration_sec", color="Type", 
                         log_y=True, title=f"Események hossza - {prefix}",
                         labels={'Duration_sec': 'Hossz (mp)'}, barmode='stack')
    save_responsive_html(fig2, f"{VIS_FOLDER}/{prefix}_duration.html")


    fig3 = px.scatter(df.sample(min(len(df), 50000)), x="Packet speed", y="Data speed", color="Type",
                        title=f"Packet vs Data Speed - {prefix}", opacity=0.4)
    save_responsive_html(fig3, f"{VIS_FOLDER}/{prefix}_traffic.html")


    fig4 = px.violin(df, y="Source IP count", x="Type", color="Type", box=True,
                       title=f"Forrás IP diverzitás - {prefix}")
    save_responsive_html(fig4, f"{VIS_FOLDER}/{prefix}_source.html")

    num_cols = ['Packet speed', 'Data speed', 'Avg packet len', 'Source IP count', 'Duration_sec']
    corr = df[num_cols].corr()
    fig5 = px.imshow(corr, text_auto=True, title=f"Jellemzők korrelációja - {prefix}")
    save_responsive_html(fig5, f"{VIS_FOLDER}/{prefix}_corr.html")

    fig6 = px.bar(df.groupby(['Attack code', 'Type'], observed=True).size().reset_index(name='C'),
                    x="Attack code", y="C", color="Type", title=f"Attack Code típusonként - {prefix}")
    save_responsive_html(fig6, f"{VIS_FOLDER}/{prefix}_codes.html")

all_dfs = []
for s in SETS:
    data = load_and_clean(s)
    if data is not None:
        generate_suite(data, s)
        all_dfs.append(data)

if all_dfs:
    combined = pd.concat(all_dfs, ignore_index=True)
    generate_suite(combined, "Combined")
    print("Minden ábra sikeresen legenerálva a /vis mappába!")