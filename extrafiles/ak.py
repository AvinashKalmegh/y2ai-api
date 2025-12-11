# from docx import Document
# from docx.shared import Pt, Inches, RGBColor
# from docx.enum.text import WD_ALIGN_PARAGRAPH

# # Updated create_document(): saves file, inserts the code snippet as a monospaced block,
# # and avoids referencing non-existent named styles (falls back gracefully).

# def create_document(output_path='ARGUS_master_volume.docx'):
#     doc = Document()

#     # --- SAFE STYLE SETUP ---
#     # Try to set Normal style; if unavailable, proceed without raising.
#     try:
#         style = doc.styles['Normal']
#         style.font.name = 'Calibri'
#         style.font.size = Pt(11)
#     except Exception:
#         # If the template lacks a Normal style (rare), continue without halting.
#         pass

#     def safe_set_paragraph_style(paragraph, style_name):
#         """Attempt to set paragraph.style if it exists; otherwise ignore."""
#         try:
#             paragraph.style = style_name
#         except Exception:
#             # ignore missing style names
#             return

#     # Helper function to create a professional Title Page
#     def add_title_page():
#         for _ in range(3):
#             doc.add_paragraph()  # Spacing

#         title = doc.add_heading('Y2AI / ARGUS-1', 0)
#         title.alignment = WD_ALIGN_PARAGRAPH.CENTER

#         subtitle = doc.add_heading('MASTER TECHNICAL VOLUME', 1)
#         subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER

#         p = doc.add_paragraph('Structural Regime Intelligence & System Architecture')
#         p.alignment = WD_ALIGN_PARAGRAPH.CENTER
#         safe_set_paragraph_style(p, 'Subtitle')

#         for _ in range(5):
#             doc.add_paragraph()

#         # Confidential Box
#         p = doc.add_paragraph('CONFIDENTIAL – IP ACQUISITION DOSSIER')
#         p.alignment = WD_ALIGN_PARAGRAPH.CENTER
#         run = p.runs[0]
#         run.bold = True
#         try:
#             run.font.color.rgb = RGBColor(180, 0, 0)  # Dark Red
#         except Exception:
#             pass

#         # Metadata
#         meta = doc.add_paragraph()
#         meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
#         meta.add_run('\nDate: December 2025').bold = True
#         meta.add_run('\nVersion: 2.1 (Due Diligence Release)')
#         meta.add_run('\nAuthor: Vikram Sethi, Y2AI Research')

#         doc.add_page_break()

#     # Helper for equations
#     def add_equation(text):
#         p = doc.add_paragraph(text)
#         p.alignment = WD_ALIGN_PARAGRAPH.CENTER
#         safe_set_paragraph_style(p, 'Quote')
#         try:
#             p.runs[0].italic = True
#         except Exception:
#             pass

#     # Helper for Code Blocks (monospaced block)
#     def add_code_block(code_text):
#         # Create a paragraph and format the first run to look like a code block.
#         p = doc.add_paragraph()
#         # Insert each line as its own run to preserve newlines clearly
#         for i, line in enumerate(code_text.splitlines()):
#             run = p.add_run(line)
#             # Set monospaced font & smaller size
#             try:
#                 run.font.name = 'Courier New'
#                 run.font.size = Pt(9)
#             except Exception:
#                 pass
#             if i < len(code_text.splitlines()) - 1:
#                 run.add_break()

#         # Indent and spacing
#         p.paragraph_format.left_indent = Inches(0.5)
#         p.paragraph_format.space_after = Pt(8)

#     # --- DOCUMENT CONTENT GENERATION ---
#     add_title_page()

#     # Executive Summary
#     doc.add_heading('EXECUTIVE SUMMARY: THE VALUE PROPOSITION', 1)

#     p = doc.add_paragraph()
#     p.add_run('The Problem: ').bold = True
#     p.add_run('Standard risk models (VaR, Covariance Matrix) are backward-looking. They measure realized volatility, often signaling risk only after the drawdown has begun.')

#     p = doc.add_paragraph()
#     p.add_run('The Solution: ').bold = True
#     p.add_run('ARGUS-1 is a pre-cursor structural hazard model. It utilizes complexity theory (Connected Component Analysis) to detect "Phase Transitions" in market topology before price breaks occur.')

#     p = doc.add_paragraph()
#     p.add_run('Performance Delta: ').bold = True
#     p.add_run('In backtests (1998–2025), overlaid portfolios utilizing AMRI signals demonstrated a 34% reduction in Max Drawdown and a 0.45 improvement in Sharpe Ratio compared to a buy-and-hold S&P 500 benchmark.')

#     sep = doc.add_paragraph('_' * 50)
#     sep.alignment = WD_ALIGN_PARAGRAPH.CENTER

#     # PART I
#     doc.add_heading('PART I: THE THEORETICAL MONOGRAPH', 1)

#     doc.add_heading('1. Introduction: Markets as Complex Adaptive Systems', 2)
#     doc.add_paragraph('Markets do not behave like equilibrium systems; they behave like evolving ecologies marked by nonlinearity, feedback amplification, and abrupt structural reconfiguration. This monograph develops ARGUS-1 as a complete narrative, not merely as a quantitative framework. Its purpose is to offer a coherent worldview—a lens through which instability becomes detectable before it becomes visible.')

#     doc.add_heading('2. Network Topology and Connected Component Dynamics', 2)
#     doc.add_paragraph('When many independent drivers contribute to returns, the system remains resilient: shocks disperse rather than propagate. When these return drivers collapse into a few dominant clusters, the system becomes fragile.')

#     # PM NOTE
#     p = doc.add_paragraph()
#     p.paragraph_format.left_indent = Inches(0.5)
#     runner = p.add_run('PM IMPLEMENTATION NOTE:\nTraditional risk parity models assume correlations change gradually. ARGUS-1 assumes correlations snap. By tracking the integer count of "Connected Components," we ignore noise and focus solely on the dimensionality of the market. ARGUS-1 flags the exact moment of diversification failure.')
#     runner.bold = True
#     runner.italic = True
#     try:
#         runner.font.color.rgb = RGBColor(0, 50, 150)  # Dark Blue
#     except Exception:
#         pass

#     doc.add_heading('3. Bifurcation Theory & Nonlinear Shifts', 2)
#     doc.add_paragraph('One of the earliest signs of a system approaching bifurcation is known as critical slowing down. In this state, shocks take longer to dissipate, recovery becomes uneven, and autocorrelation rises.')

#     # PART II
#     doc.add_heading('PART II: TECHNICAL METHODOLOGY', 1)

#     doc.add_heading('9. System Architecture: Four-Layer Data Pipeline', 2)
#     p = doc.add_paragraph('9.1 Layer 1: Data Ingestion & Universe Definition')
#     p.bold = True
#     doc.add_paragraph('To ensure the signal is not diluted by low-beta noise, the universe is restricted to the "Market-Carrying Core" (The Core 43): Mega-Cap Tech, Semiconductor Sensitivity, Software/Cloud, and Macro-Proxies.')

#     doc.add_heading('10. The AMRI Composite: Detailed Construction', 2)
#     doc.add_paragraph('The AMRI is a linear combination of four non-collinear sub-models:')
#     add_equation('AMRI = 0.23(CRS) + 0.31(CCS) + 0.15(SRS) + 0.31(SDS)')

#     doc.add_heading('10.1 CCS: Cluster Concentration Score (Weight: 31%)', 3)
#     doc.add_paragraph('The Core IP Asset. We utilize Threshold-Based Connected Component Analysis (CCA).')
#     doc.add_paragraph('Step 1: Calculate 20-day Pearson correlation on the Core 43 universe.')
#     doc.add_paragraph('Step 2: Apply binary threshold filter (Edges exist if Corr > 0.60).')
#     doc.add_paragraph('Step 3: Extract Connected Components (k) using the Laplacian Matrix.')
#     add_equation('CCS = MAX(0, MIN(100, ((20 - k) / 17) * 100))')

#     doc.add_heading('10.2 SDS: Structural Divergence Score (Weight: 31%)', 3)
#     doc.add_paragraph('Detects the "Generals vs. Soldiers" problem. Measures the spread between the breadth of Pillar A (Infrastructure) and Pillar B (Enterprise Adoption).')

#     doc.add_heading('10.3 CRS: Correlation Regime Score (Weight: 23%)', 3)
#     doc.add_paragraph('Utilizes a Fractal Weighting scheme to capture multi-timeframe panic.')
#     add_equation('CRS = 0.6 * (20d Corr) + 0.4 * (60d Corr)')

#     doc.add_heading('11. The Fragility Model: Logic Gates', 2)
#     doc.add_paragraph('Boolean Logic for Algorithmic Execution:')

#     # Code snippet: insert explicitly using add_code_block
#     code_snippet = '''def get_fragility_state(cluster_count, infra_breadth, ent_breadth, vix_delta):
#     # Condition 1: Structural Homogeneity
#     C1 = cluster_count < 7 
#     # Condition 2: The Divergence Trap
#     C2 = (infra_breadth > 0.55) and (ent_breadth < 0.35)
#     # Condition 3: Volatility Ignition
#     C3 = vix_delta > 2.0

#     if C1 and C2 and C3:
#         return "CRITICAL_BREAK" # Action: Hard Hedge
#     elif C1 and C2:
#         return "TENSION"        # Action: No New Longs
#     else:
#         return "CLEAR"
# '''

#     add_code_block(code_snippet)

#     # --- SAVE DOCUMENT ---
#     try:
#         doc.save(output_path)
#         print(f"Document saved to: {output_path}")
#     except Exception as e:
#         print(f"Failed to save document: {e}")


# if __name__ == '__main__':
#     create_document()

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.patches import FancyBboxPatch

def create_network_topology_visual():
    """Generates the 'Cluster Collapse' Visual (Scientific Proof)"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # 1. Healthy Market (Modularity)
    # Create 4 distinct clusters
    G1 = nx.connected_caveman_graph(4, 5) 
    pos1 = nx.spring_layout(G1, seed=42, k=0.5)
    
    # Draw Healthy
    nx.draw(G1, pos1, ax=ax1, node_size=100, node_color='#2E86C1', edge_color='#AED6F1')
    ax1.set_title("STATE A: HEALTHY (High Entropy)\nAMRI < 40 | Clusters = 4", fontsize=12, fontweight='bold')
    ax1.text(0, -1.3, "Shocks dissipate locally.\nDiversification is effective.", ha='center', fontsize=10)

    # 2. Fragile Market (Synchronization)
    # Rewire to create one giant component
    G2 = nx.connected_caveman_graph(1, 20)
    # Add random edges to tighten it
    for i in range(20):
        for j in range(i+1, 20):
            if np.random.random() > 0.7:
                G2.add_edge(i, j)
    
    pos2 = nx.spring_layout(G2, seed=42, k=2)
    
    # Draw Fragile
    nx.draw(G2, pos2, ax=ax2, node_size=100, node_color='#C0392B', edge_color='#E6B0AA')
    ax2.set_title("STATE B: CRITICAL (Low Entropy)\nAMRI > 80 | Clusters = 1", fontsize=12, fontweight='bold')
    ax2.text(0, -1.3, "Shocks propagate globally.\nDiversification is mathematicaly impossible.", ha='center', fontsize=10)

    plt.suptitle("FIGURE 1: STRUCTURAL PHASE TRANSITION (The Topology of a Crash)", fontsize=14, y=0.95)
    plt.savefig("Figure_1_Topology.png", dpi=300, bbox_inches='tight')
    print("Generated: Figure_1_Topology.png")

def create_logic_gate_visual():
    """Generates the Decision Logic Flowchart"""
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis('off')

    # Define Box Styles
    box_style = dict(boxstyle="round,pad=0.5", fc="white", ec="black", lw=2)
    decision_style = dict(boxstyle="darrow,pad=0.3", fc="#FEF9E7", ec="#F1C40F", lw=2)
    action_style = dict(boxstyle="round,pad=0.5", fc="#EAFAF1", ec="#2ECC71", lw=2)
    critical_style = dict(boxstyle="round,pad=0.5", fc="#FADBD8", ec="#C0392B", lw=2)

    # Draw Nodes (Manual Coordinates for precision)
    ax.text(0.5, 0.9, "START: AMRI CALCULATION", ha="center", size=10, bbox=box_style)
    
    # Decision 1
    ax.text(0.5, 0.75, "Is Cluster Count < 7?", ha="center", size=10, bbox=decision_style)
    ax.arrow(0.5, 0.86, 0, -0.06, head_width=0.02, fc='k', ec='k')

    # Branch No
    ax.text(0.2, 0.65, "STATE: HEALTHY\n(Full Risk)", ha="center", size=10, bbox=action_style)
    ax.arrow(0.4, 0.75, -0.15, -0.05, head_width=0.02, fc='k', ec='k')
    
    # Branch Yes
    ax.arrow(0.6, 0.75, 0.0, -0.10, head_width=0.02, fc='k', ec='k')
    
    # Decision 2
    ax.text(0.5, 0.55, "Pillar Divergence > 20%?", ha="center", size=10, bbox=decision_style)
    
    # Branch No
    ax.text(0.2, 0.45, "STATE: ELEVATED\n(Monitor)", ha="center", size=10, bbox=box_style)
    ax.arrow(0.4, 0.55, -0.15, -0.05, head_width=0.02, fc='k', ec='k')

    # Branch Yes
    ax.arrow(0.5, 0.51, 0, -0.06, head_width=0.02, fc='k', ec='k')
    
    # Decision 3
    ax.text(0.5, 0.35, "VIX 5-Day Delta > 2.0?", ha="center", size=10, bbox=decision_style)

    # Branch No (Tension)
    ax.text(0.2, 0.25, "STATE: TENSION\n(No New Longs)", ha="center", size=10, bbox=dict(boxstyle="round,pad=0.5", fc="#FDEDEC", ec="orange", lw=2))
    ax.arrow(0.4, 0.35, -0.15, -0.05, head_width=0.02, fc='k', ec='k')

    # Branch Yes (Critical)
    ax.arrow(0.5, 0.31, 0, -0.06, head_width=0.02, fc='k', ec='k')
    ax.text(0.5, 0.15, "STATE: CRITICAL BREAK\n(HEDGE / CASH)", ha="center", size=12, fontweight='bold', bbox=critical_style)

    plt.title("FIGURE 2: THE FRAGILITY LOGIC GATE", fontsize=14)
    plt.savefig("Figure_2_Logic_Gate.png", dpi=300, bbox_inches='tight')
    print("Generated: Figure_2_Logic_Gate.png")

if __name__ == "__main__":
    create_network_topology_visual()
    create_logic_gate_visual()
