import graphviz

def create_conceptual_diagram():
    """Create a conceptual diagram of the RL-based EV charging system."""
    dot = graphviz.Digraph('RL_Charging_System', 
                          comment='Conceptual Diagram of RL-based EV Charging System',
                          format='png')
    
    # Set diagram attributes for better visualization
    dot.attr(rankdir='LR', splines='ortho')
    dot.attr('node', shape='box', style='rounded,filled', fontname='Arial',
             fontsize='12', margin='0.3,0.2')
    
    # Input cluster
    with dot.subgraph(name='cluster_0') as inputs:
        inputs.attr(label='Inputs', style='rounded', color='lightblue', bgcolor='azure')
        inputs.node('battery', 'Battery State\n- Charge Level\n- Capacity\n- Target SoC', fillcolor='lightblue')
        inputs.node('grid', 'Grid Status\n- Current Load\n- Peak Hours\n- Capacity', fillcolor='lightblue')
        inputs.node('price', 'Price Signals\n- Time-of-Use\n- Dynamic Rates\n- Grid Demand', fillcolor='lightblue')
    
    # RL Agent cluster
    with dot.subgraph(name='cluster_1') as agent:
        agent.attr(label='RL Agent', style='rounded', color='lightgreen', bgcolor='mintcream')
        agent.node('state', 'State Processing', fillcolor='palegreen')
        agent.node('policy', 'Policy Network\n(PPO)', fillcolor='palegreen')
        agent.node('action', 'Action Selection', fillcolor='palegreen')
    
    # Output cluster
    with dot.subgraph(name='cluster_2') as outputs:
        outputs.attr(label='Outputs', style='rounded', color='coral', bgcolor='seashell')
        outputs.node('schedule', 'Charging Schedule\n- Start/Stop Times\n- Power Levels', fillcolor='lightsalmon')
        outputs.node('savings', 'Performance Metrics\n- Cost Savings\n- Grid Stability\n- Wait Times', fillcolor='lightsalmon')
    
    # Connect nodes
    # Inputs to State Processing
    dot.edge('battery', 'state')
    dot.edge('grid', 'state')
    dot.edge('price', 'state')
    
    # RL Agent internal flow
    dot.edge('state', 'policy')
    dot.edge('policy', 'action')
    
    # Action to Outputs
    dot.edge('action', 'schedule')
    dot.edge('schedule', 'savings')
    
    # Save the diagram
    dot.render('docs/images/conceptual_diagram', cleanup=True)

if __name__ == "__main__":
    create_conceptual_diagram() 