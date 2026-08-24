import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
import io
from matplotlib.colors import ListedColormap
from matplotlib.colors import ListedColormap, Normalize


# --- SETTINGS
BASE = "https://raw.githubusercontent.com/MeijerW/ProteomeUI/main/Datafiles/"

st.title("Somitogenesis Gene Expression Explorer")

# Set Seaborn style
sns.set(style="whitegrid")
sns.set_context("paper")

# Color palettes
rna_palette = {
    "posterior": "#d5af34",
    "anterior": "#f9d777",
    "somite": "#f9e7b7"
}
prot_palette = {
    "posterior": "#8281be",
    "anterior": "#b2b2d9",
    "somite": "#d7d6ea"
}
# Load data once, use throughout
RNA_URL = BASE + "RNA_preprocessed.csv"
PROT_URL = BASE + "Protein_preprocessed.csv"
            

RNA_FILES = {
    "Anterior": "RNAseq_Spatiotemporal_anterior.csv",
    "Posterior": "RNAseq_Spatiotemporal_posterior.csv",
    "Somite": "RNAseq_Spatiotemporal_somite.csv",
}
PROT_FILES = {
    "Anterior": "Proteomics_Spatiotemporal_anterior.csv",
    "Posterior": "Proteomics_Spatiotemporal_posterior.csv",
    "Somite": "Proteomics_Spatiotemporal_somite.csv",
}


#---FUNCTIONS
@st.cache_data
def load_data():
    rna = pd.read_csv(RNA_URL)
    prot = pd.read_csv(PROT_URL)
    rna['Type'] = 'RNA'
    prot['Type'] = 'Protein'
    return rna, prot

@st.cache_data
def load_spatiotemporal_data():
    def load_files(file_dict):
        dfs = {}
        for region, filename in file_dict.items():
            df = pd.read_csv(BASE + filename, sep=',')  # Although file says .tsv, it's actually CSV
            df.set_index('ID', inplace=True)
            dfs[region] = df
        return dfs

    return load_files(RNA_FILES), load_files(PROT_FILES)

def prepare_long_df(df_dict, gene, datatype):
    gene = gene.strip().lower()
    all_data = []
    for region, df in df_dict.items():
        df.index = df.index.astype(str).str.strip().str.lower()

        if gene not in df.index:
            continue

        row = df.loc[gene]
        pval = row["P_VALUE"] if "P_VALUE" in row.index else np.nan  # grab p-value if present

        expression = row.filter(like='TP_')
        melted = expression.reset_index()
        melted.columns = ['Condition', 'Expression']

        melted['Time'] = melted['Condition'].str.extract(r'TP_(\d+)_')[0]
        melted['Rep'] = melted['Condition'].str.extract(r'REP_(\d+)')[0]
        melted['Time'] = pd.Categorical(
            melted['Time'],
            categories=['30', '60', '90', '120'],
            ordered=True
        )
        melted['Region'] = region
        melted['Type'] = datatype
        melted['P_VALUE'] = pval  # attach p-value to all rows

        all_data.append(melted)

    return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()

def plot_expression_grid(df, gene_name, region):
    sns.set(style="whitegrid")
    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    for row, datatype in enumerate(["RNA", "Protein"]):
        ax = axes[row]
        sub_df = df[(df['Region'] == region) & (df['Type'] == datatype)]
        if sub_df.empty:
            ax.axis("off")
            ax.set_title(f"{datatype} data missing", fontsize=16, color="red")
            continue

        # Palette
        color = rna_palette[region.lower()] if datatype == "RNA" else prot_palette[region.lower()]

        # Boxplot
        sns.boxplot(
            data=sub_df,
            x="Time", y="Expression",
            color=color, ax=ax,
            order=['30', '60', '90', '120'],
            fliersize=0, width=0.6
        )
        # Overlay replicates
        sns.stripplot(
            data=sub_df,
            x="Time", y="Expression",
            color="black", ax=ax,
            order=['30', '60', '90', '120'],
            size=3, jitter=True
        )

        ax.set_title(f"{datatype} Expression", fontsize=18)
        ax.set_xlabel("Time (min)", fontsize=12)
        ax.set_ylabel("Expression", fontsize=12)

        # Add p-value annotation
        pval = sub_df['P_VALUE'].iloc[0] if 'P_VALUE' in sub_df.columns else np.nan
        if not pd.isna(pval):
            # Show p-value in plain decimal notation with up to 6 significant digits
            pval_str = f"{pval:.6f}".rstrip("0").rstrip(".")  # trim trailing zeros/dot
            ax.text(
                0.98, 0.95,
                f"p = {pval_str}",
                ha="right", va="top",
                transform=ax.transAxes,
                fontsize=12, color="red", fontweight="bold"
            )

    fig.suptitle(f"{region} Spatiotemporal Expression of {gene_name}", fontsize=20, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig

def prepare_heatmap_matrix(df_dict, gene_list, region):
    """
    Returns a DataFrame indexed by gene, columns = timepoints (30,60,90,120)
    with the mean across replicates for that region and those genes.
    """
    # Get the dataframe for the chosen region
    df = df_dict[region].copy()
    # Normalize index to lowercase (so gene_list lowercased will match)
    df.index = df.index.astype(str).str.strip().str.lower()

    # Only keep requested genes
    gene_idx = [g for g in gene_list if g in df.index]
    if not gene_idx:
        return pd.DataFrame()  # nothing found

    subset = df.loc[gene_idx, :]
    # keep only TP_ columns
    expr = subset.filter(like="TP_").copy()

    # reset index and ensure first column is gene id
    expr = expr.reset_index()
    expr = expr.rename(columns={expr.columns[0]: "ID"})

    # melt and extract times
    melted = expr.melt(id_vars="ID", var_name="Condition", value_name="Expression")
    melted["Time"] = melted["Condition"].str.extract(r"TP_(\d+)_")[0]
    melted["Time"] = pd.Categorical(melted["Time"], categories=["30", "60", "90", "120"], ordered=True)

    # average replicates per gene × time
    avg = melted.groupby(["ID", "Time"])["Expression"].mean().unstack("Time")

    # keep clean names
    avg.index.name = None
    avg.columns.name = None
    return avg

def zscore_matrix(df):
    return df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1), axis=0)

def prepare_pval_matrix(df_dict, gene_list, region):
    df = df_dict[region].copy()
    df.index = df.index.astype(str).str.strip().str.lower()
    gene_idx = [g for g in gene_list if g in df.index]
    if not gene_idx:
        return pd.DataFrame()
    pvals = df.loc[gene_idx, "P_VALUE"]
    return pd.DataFrame({"p-value": pvals})

def plot_heatmaps(rna_matrix, prot_matrix, rna_pvals, prot_pvals, region, gene_list):
    # Gene order is taken from RNA if available, else protein
    if not rna_matrix.empty:
        genes_order = list(rna_matrix.index)
    else:
        genes_order = list(prot_matrix.index)

    # Reindex all matrices consistently
    rna_matrix = rna_matrix.reindex(genes_order)
    prot_matrix = prot_matrix.reindex(genes_order)
    rna_pvals = rna_pvals.reindex(genes_order)
    prot_pvals = prot_pvals.reindex(genes_order)

    # Z-score normalization
    if not rna_matrix.empty:
        rna_matrix = zscore_matrix(rna_matrix)
    if not prot_matrix.empty:
        prot_matrix = zscore_matrix(prot_matrix)

    # Figure with 2 rows × 2 cols
    fig = plt.figure(figsize=(14, max(3, len(genes_order) * 0.4)))
    gs = gridspec.GridSpec(2, 2, width_ratios=[3, 1], height_ratios=[1, 1], wspace=0.05, hspace=0.25)

    # --- RNA expression ---
    ax1 = fig.add_subplot(gs[0, 0])
    if not rna_matrix.empty:
        sns.heatmap(rna_matrix, cmap="vlag", ax=ax1, cbar=True,
                    center=0, vmin=-2, vmax=2, yticklabels=True)
        ax1.set_title("RNA Expression (z-score)")
        ax1.set_xlabel("Time (min)")
        ax1.set_ylabel("Genes")
    else:
        ax1.axis("off")
        ax1.set_title("No RNA data")

    # --- RNA p-values ---
    ax2 = fig.add_subplot(gs[0, 1], sharey=ax1)
    if not rna_pvals.empty:
        sns.heatmap(rna_pvals, cmap="Reds_r", annot=True, fmt=".5f",
                    cbar=False, ax=ax2, yticklabels=False)
        ax2.set_title("RNA p-values")
        ax2.set_xlabel("")
        ax2.set_ylabel("")
    else:
        ax2.axis("off")
        ax2.set_title("No RNA p-values")

    # --- Protein expression ---
    ax3 = fig.add_subplot(gs[1, 0])
    if not prot_matrix.empty:
        sns.heatmap(prot_matrix, cmap="vlag", ax=ax3, cbar=True,
                    center=0, vmin=-2, vmax=2, yticklabels=True)
        ax3.set_title("Protein Expression (z-score)")
        ax3.set_xlabel("Time (min)")
        ax3.set_ylabel("Genes")
    else:
        ax3.axis("off")
        ax3.set_title("No Protein data")

    # --- Protein p-values ---
    ax4 = fig.add_subplot(gs[1, 1], sharey=ax3)
    if not prot_pvals.empty:
        sns.heatmap(prot_pvals, cmap="Reds_r", annot=True, fmt=".5f",
                    cbar=False, ax=ax4, yticklabels=False)
        ax4.set_title("Protein p-values")
        ax4.set_xlabel("")
        ax4.set_ylabel("")
    else:
        ax4.axis("off")
        ax4.set_title("No Protein p-values")

    fig.suptitle(f"{region} Spatiotemporal Heatmaps", fontsize=18, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def add_cbar_row(fig, ims, labels, height=0.03, pad=0.05):
    """
    Add a row of horizontal colorbars beneath the figure, one for each heatmap.
    - fig: the matplotlib figure
    - ims: list of mappable objects (e.g. im from sns.heatmap)
    - labels: list of labels for the colorbars
    - height: relative height of each cbar (fraction of figure height)
    - pad: vertical padding between heatmaps and cbar row
    """
    n = len(ims)
    fig_w, fig_h = fig.get_size_inches()

    # get bottom of heatmaps
    bbox = ims[0].axes.get_position()
    y_bottom = bbox.y0

    # create equally spaced cax slots under each heatmap
    for i, (im, label) in enumerate(zip(ims, labels)):
        ax = im.axes
        bbox = ax.get_position()
        x0, x1 = bbox.x0, bbox.x1
        cax = fig.add_axes([x0, y_bottom - pad - height, x1 - x0, height])
        cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
        cbar.set_label(label)
        cbar.outline.set_visible(False)


# Top-level tabs
main_tab1, main_tab2 = st.tabs(["Spatial Viewer", "Spatiotemporal Viewer"])

# ────────── Spatial Viewer ──────────
with main_tab1:
    subtab1, subtab2 = st.tabs(["Single Gene", "Heatmap (Multiple Genes)"])

    #Load spatial data directly
    rna_df, prot_df = load_data()
    rna_df['group'] = rna_df['group'].str.lower()
    prot_df['group'] = prot_df['group'].str.lower()
    
    # Single Gene Boxplot
    with subtab1:
        st.markdown("### Single Gene Spatial Expression")
        st.markdown("Compare spatial expression of a gene across RNA and Protein levels.")
        gene_input = st.text_input("Enter gene name (e.g., tbx6):", value="")

        if gene_input:
            rna_plot = rna_df[rna_df["Gene"].str.lower() == gene_input.lower()]
            prot_plot = prot_df[prot_df["Gene"].str.lower() == gene_input.lower()]

            if rna_plot.empty and prot_plot.empty:
                st.warning(f"No data found for gene '{gene_input}'.")
            else:
                fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

                sns.boxplot(
                    data=rna_plot,
                    x="group",
                    y="Z-score",
                    order=["posterior", "anterior", "somite"],
                    palette=rna_palette,
                    ax=axes[0]
                )
                axes[0].set_title("RNA Expression", fontsize=26, fontweight='bold')
                axes[0].set_xlabel("")
                axes[0].set_ylabel("Z-score", fontsize=16)
                axes[0].tick_params(axis='x', labelsize=16)

                sns.boxplot(
                    data=prot_plot,
                    x="group",
                    y="Z-score",
                    order=["posterior", "anterior", "somite"],
                    palette=prot_palette,
                    ax=axes[1]
                )
                axes[1].set_title("Protein Expression", fontsize=26, fontweight='bold')
                axes[1].set_xlabel("")
                axes[1].set_ylabel("")
                axes[1].tick_params(axis='x', labelsize=16)

                plt.tight_layout()
                st.pyplot(fig)
                # Save figure to a BytesIO buffer
                buf = io.BytesIO()
                fig.savefig(buf, format="png", bbox_inches="tight", dpi=300)
                buf.seek(0)
                
                # Add download button
                st.download_button(
                    label="📥 Download this figure as PNG",
                    data=buf,
                    file_name= (gene_input + "_spatial_boxplot.png"),
                    mime="image/png"
)
    # Heatmap Tab 
        with subtab2:
            st.markdown("### Spatial Heatmap for Multiple Genes")
            st.markdown("Enter multiple gene names separated by commas (e.g., `tbx6, msgn1, dlc`). Capitalization does not matter.")
        
            gene_input = st.text_input("Genes for heatmap:", value="")
        
            if gene_input:
                gene_list = [g.strip().lower() for g in gene_input.split(",") if g.strip()]
        
                rna_subset = rna_df[rna_df["Gene"].str.lower().isin(gene_list)]
                prot_subset = prot_df[prot_df["Gene"].str.lower().isin(gene_list)]
        
                def prepare_avg(df):
                    grouped = df.groupby(['Gene', 'group'])['Z-score'].mean()
                    unstacked = grouped.unstack(fill_value=np.nan)
                    if isinstance(unstacked, pd.Series):
                        unstacked = unstacked.to_frame().T
                    return unstacked
        
                rna_avg = prepare_avg(rna_subset)
                prot_avg = prepare_avg(prot_subset)
        
                expected_regions = ["posterior", "anterior", "somite"]
                rna_avg = rna_avg.reindex(columns=expected_regions)
                prot_avg = prot_avg.reindex(columns=expected_regions)
        
                # Only keep genes with any data in RNA
                rna_avg = rna_avg.dropna(how='all')
                prot_avg = prot_avg.reindex(rna_avg.index).sort_index()
                
                # Check missing genes
                # Normalize index to lowercase for comparison
                found_genes_lower = {g.lower() for g in rna_avg.index.union(prot_avg.index)}
                genes_not_found = [g for g in gene_list if g not in found_genes_lower]
                
                if genes_not_found:
                    st.warning(f"The following genes were not found in either RNA or Protein datasets and will not be shown: {', '.join(genes_not_found)}")

                if rna_avg.empty:
                    st.warning("None of the entered genes were found in the RNA dataset.")
                else:
                    # Get clustering order
                    g = sns.clustermap(
                        rna_avg,
                        cmap="viridis",
                        row_cluster=True,
                        col_cluster=False,
                        cbar_pos=None,
                        figsize=(1, 1)  # dummy
                    )
                    plt.close()
        
                    gene_order = [rna_avg.index[i] for i in g.dendrogram_row.reordered_ind]
                    rna_ordered = rna_avg.loc[gene_order]
                    prot_ordered = prot_avg.loc[gene_order]
        
                    # Separate vmin/vmax for each
                    vmin_rna, vmax_rna = np.nanmin(rna_ordered.values), np.nanmax(rna_ordered.values)
                    vmin_prot, vmax_prot = np.nanmin(prot_ordered.values), np.nanmax(prot_ordered.values)
        
                    # Layout
                    fig = plt.figure(figsize=(12, len(gene_order) * 0.4 + 3))
                    gs = gridspec.GridSpec(2, 2, height_ratios=[20, 1], width_ratios=[1, 1], hspace=0.4, wspace=0.05)
        
                    ax1 = fig.add_subplot(gs[0, 0])
                    ax2 = fig.add_subplot(gs[0, 1], sharey=ax1)
                    cax1 = fig.add_subplot(gs[1, 0])
                    cax2 = fig.add_subplot(gs[1, 1])
        
                    sns.heatmap(
                        rna_ordered,
                        cmap="viridis",
                        ax=ax1,
                        cbar=False,
                        vmin=vmin_rna,
                        vmax=vmax_rna,
                        yticklabels=True
                    )
                    ax1.set_title("RNA Expression (clustered)", fontsize=14)
                    ax1.set_xlabel("")
                    ax1.set_ylabel("")
                    ax1.set_yticklabels(rna_ordered.index, rotation=0)
                    
                    sns.heatmap(
                        prot_ordered,
                        cmap="viridis",
                        ax=ax2,
                        cbar=False,
                        vmin=vmin_prot,
                        vmax=vmax_prot,
                        yticklabels=True  # <--- Changed here
                    )
                    ax2.set_title("Protein Expression", fontsize=14)
                    ax2.set_xlabel("")
                    ax2.set_ylabel("")
                    ax2.yaxis.tick_right()
                    ax2.yaxis.set_label_position("right")
                    ax2.set_yticklabels(prot_ordered.index, rotation=0)
        
                    # Colorbars
                    sm_rna = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin=vmin_rna, vmax=vmax_rna))
                    sm_rna.set_array([])
                    cbar1 = fig.colorbar(sm_rna, cax=cax1, orientation='horizontal')
                    cbar1.set_label("Z-score (RNA)")
        
                    sm_prot = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin=vmin_prot, vmax=vmax_prot))
                    sm_prot.set_array([])
                    cbar2 = fig.colorbar(sm_prot, cax=cax2, orientation='horizontal')
                    cbar2.set_label("Z-score (Protein)")
        
                    st.pyplot(fig)
                    # Save figure to a BytesIO buffer
                    buf = io.BytesIO()
                    fig.savefig(buf, format="png", bbox_inches="tight", dpi=300)
                    buf.seek(0)
                    
                    # Add download button
                    st.download_button(
                        label="📥 Download Heatmap as PNG",
                        data=buf,
                        file_name="spatial_heatmap.png",
                        mime="image/png"
                    )

# ────────── Spatiotemporal Viewer ──────────
with main_tab2:
    subtab3, subtab4 = st.tabs(["Single Gene", "Heatmap (Multiple Genes)"])

    with subtab3:
        # st.markdown("### Single gene dynamic expression")
        # # load dicts once
        # rna_dict, prot_dict = load_spatiotemporal_data()
    
        # # Unique keys for widgets
        # gene_input = st.text_input("Enter gene name (single):", value="", key="st_spatio_single_gene")
        # region_choice = st.selectbox(
        #     "Select region (single)", ["Posterior", "Anterior",  "Somite"],
        #     index=0, key="st_spatio_single_region"
        # )
    
        # if gene_input:
        #     # prepare long format for the selected region only
        #     # note: prepare_long_df expects a dict of {region: df}
        #     rna_long = prepare_long_df({region_choice: rna_dict[region_choice]}, gene_input, "RNA")
        #     prot_long = prepare_long_df({region_choice: prot_dict[region_choice]}, gene_input, "Protein")
        #     combined_df = pd.concat([rna_long, prot_long], ignore_index=True)
    
        #     if combined_df.empty:
        #         st.warning(f"Gene '{gene_input}' not found in {region_choice} datasets.")
        #     else:
        #         fig = plot_expression_grid(combined_df, gene_input, region_choice)
        #         st.pyplot(fig)
    
        #         # unique download key and filename
        #         buf = io.BytesIO()
        #         fig.savefig(buf, format="png", bbox_inches="tight", dpi=300)
        #         buf.seek(0)
        #         st.download_button(
        #             label="📥 Download this figure as PNG (single)",
        #             data=buf,
        #             file_name=f"{gene_input}_{region_choice}_spatiotemporal_expression.png",
        #             mime="image/png",
        #             key=f"download_spatiotemp_single_{gene_input}_{region_choice}"
        #         )

        with subtab3:
            st.markdown("### Single gene dynamic expression (Z-scored)")
            # load dicts once
            rna_dict, prot_dict = load_spatiotemporal_data()
        
            # Unique keys for widgets
            gene_input = st.text_input("Enter gene name (single):", value="", key="st_spatio_single_gene")
            region_choice = st.selectbox(
                "Select region (single)", ["Posterior", "Anterior", "Somite"],
                index=0, key="st_spatio_single_region"
            )
        
            if gene_input:
                # prepare long format for the selected region only
                rna_long = prepare_long_df({region_choice: rna_dict[region_choice]}, gene_input, "RNA")
                prot_long = prepare_long_df({region_choice: prot_dict[region_choice]}, gene_input, "Protein")
        
                combined_df = pd.concat([rna_long, prot_long], ignore_index=True)
        
                if combined_df.empty:
                    st.warning(f"Gene '{gene_input}' not found in {region_choice} datasets.")
                else:
                    # --- Z-score normalization within each Type (RNA / Protein) ---
                    combined_df["Zscore"] = combined_df.groupby("Type")["Expression"].transform(
                        lambda x: (x - x.mean()) / x.std(ddof=0) if x.std(ddof=0) else 0
                    )
        
                    # Replace Expression with Zscore for plotting
                    combined_df["Expression"] = combined_df["Zscore"]
        
                    # Plot
                    fig = plot_expression_grid(combined_df, gene_input, region_choice)
                    st.pyplot(fig)
        
                    # unique download key and filename
                    buf = io.BytesIO()
                    fig.savefig(buf, format="png", bbox_inches="tight", dpi=300)
                    buf.seek(0)
                    st.download_button(
                        label="📥 Download this figure as PNG (single, Z-scored)",
                        data=buf,
                        file_name=f"{gene_input}_{region_choice}_spatiotemporal_expression_Zscore.png",
                        mime="image/png",
                        key=f"download_spatiotemp_single_{gene_input}_{region_choice}"
                    )
        
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        import matplotlib.colors as mcolors
            
        with subtab4:
            st.markdown("### Multi-gene spatiotemporal heatmaps (Z-scored)")
        
            rna_dict, prot_dict = load_spatiotemporal_data()
        
            gene_input = st.text_area("Enter gene names (comma-separated):", key="st_spatio_multi_gene")
            region_choice = st.selectbox(
                "Select region (multi)", ["Posterior", "Anterior", "Somite"],
                index=0, key="st_spatio_multi_region"
            )
        
            if gene_input:
                gene_list = [g.strip().lower() for g in gene_input.split(",") if g.strip()]
                if not gene_list:
                    st.warning("Please enter at least one gene.")
                else:
                    # --- Matrices
                    rna_matrix = prepare_heatmap_matrix(rna_dict, gene_list, region_choice)
                    prot_matrix = prepare_heatmap_matrix(prot_dict, gene_list, region_choice)
                    rna_pvals = prepare_pval_matrix(rna_dict, gene_list, region_choice)
                    prot_pvals = prepare_pval_matrix(prot_dict, gene_list, region_choice)
        
                    if rna_matrix.empty and prot_matrix.empty:
                        st.warning(f"No data found for {region_choice} in the given gene list.")
                    else:
                        # --- Ensure same gene order across RNA & Protein
                        genes_order = list(set(rna_matrix.index).union(set(prot_matrix.index)))
                        genes_order = [g for g in gene_list if g in genes_order]  # keep input order
        
                        rna_matrix = rna_matrix.reindex(genes_order)
                        prot_matrix = prot_matrix.reindex(genes_order)
                        rna_pvals = rna_pvals.reindex(genes_order)
                        prot_pvals = prot_pvals.reindex(genes_order)
        
                        # z-score
                        if not rna_matrix.empty:
                            rna_matrix = zscore_matrix(rna_matrix)
                        if not prot_matrix.empty:
                            prot_matrix = zscore_matrix(prot_matrix)
        
                        fig, axes = plt.subplots(
                            1, 4,
                            figsize=(16, max(3, len(genes_order) * 0.4)),
                            gridspec_kw={"width_ratios": [3, 1, 3, 1]}
                        )
        
                        # --- RNA expression ---
                        if not rna_matrix.empty:
                            im_rna = sns.heatmap(
                                rna_matrix, cmap="viridis", ax=axes[0],
                                center=0, vmin=-2, vmax=2,
                                cbar=False, yticklabels=True
                            )
                            axes[0].set_title("RNA Expression (z-score)")
                            axes[0].set_xlabel("Time (min)")
                            axes[0].set_ylabel("Genes")
                        else:
                            axes[0].axis("off")
                            axes[0].set_title("No RNA data")
        
                        # --- RNA p-values ---
                        if not rna_pvals.empty:
                            im_rna_p = sns.heatmap(
                                rna_pvals, cmap="RdPu_r", annot=True, fmt=".3f",
                                cbar=False, ax=axes[1], yticklabels=False
                            )
                            axes[1].set_title("RNA p-values")
                            axes[1].set_ylabel("")
                        else:
                            axes[1].axis("off")
                            axes[1].set_title("No RNA p-values")
        
                        # --- Protein expression ---
                        if not prot_matrix.empty:
                            im_prot = sns.heatmap(
                                prot_matrix, cmap="viridis", ax=axes[2],
                                center=0, vmin=-2, vmax=2,
                                cbar=False, yticklabels=False
                            )
                            axes[2].set_title("Protein Expression (z-score)")
                            axes[2].set_xlabel("Time (min)")
                            axes[2].set_ylabel("")
                        else:
                            axes[2].axis("off")
                            axes[2].set_title("No Protein data")
        
                        # --- Protein p-values ---
                        if not prot_pvals.empty:
                            im_prot_p = sns.heatmap(
                                prot_pvals, cmap="RdPu_r", annot=True, fmt=".3f",
                                cbar=False, ax=axes[3], yticklabels=False
                            )
                            axes[3].set_title("Protein p-values")
                            axes[3].set_ylabel("")
                        else:
                            axes[3].axis("off")
                            axes[3].set_title("No Protein p-values")
        
                        # --- Add colorbars below RNA/Prot expression and p-values
                        fig.subplots_adjust(bottom=0.20, wspace=0.3)
        
                        # expression cbars
                        cbar_ax1 = fig.add_axes([0.10, 0.12, 0.35, 0.03])
                        cbar_ax2 = fig.add_axes([0.55, 0.12, 0.35, 0.03])
                        if not rna_matrix.empty:
                            fig.colorbar(im_rna.collections[0], cax=cbar_ax1,
                                         orientation="horizontal", label="Z-score (RNA)")
                        if not prot_matrix.empty:
                            fig.colorbar(im_prot.collections[0], cax=cbar_ax2,
                                         orientation="horizontal", label="Z-score (Protein)")
        
                        # p-value cbars
                        cbar_ax3 = fig.add_axes([0.10, 0.05, 0.35, 0.03])
                        cbar_ax4 = fig.add_axes([0.55, 0.05, 0.35, 0.03])
                        if not rna_pvals.empty:
                            fig.colorbar(im_rna_p.collections[0], cax=cbar_ax3,
                                         orientation="horizontal", label="p-value (RNA)")
                        if not prot_pvals.empty:
                            fig.colorbar(im_prot_p.collections[0], cax=cbar_ax4,
                                         orientation="horizontal", label="p-value (Protein)")
        
                        fig.suptitle(f"{region_choice} Spatiotemporal Heatmaps", fontsize=18, fontweight="bold", y=0.99)
                        st.pyplot(fig)
        
                        # --- download
                        buf = io.BytesIO()
                        fig.savefig(buf, format="png", bbox_inches="tight", dpi=300)
                        buf.seek(0)
                        st.download_button(
                            label="📥 Download this figure as PNG (multi)",
                            data=buf,
                            file_name=f"{region_choice}_spatiotemporal_heatmaps.png",
                            mime="image/png",
                            key=f"download_spatiotemp_multi_{region_choice}"
                        )

