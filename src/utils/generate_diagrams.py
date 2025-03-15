from graphviz import Digraph
import os

def create_system_architecture():
    """Create a system architecture diagram."""
    dot = Digraph(comment='EV Charging System Architecture')
    dot.attr(rankdir='LR')
    
    # Add nodes
    with dot.subgraph(name='cluster_0') as c:
        c.attr(label='Input Layer')
        c.node('ev_data', 'EV Data\n(Arrival, Capacity)')
        c.node('grid_data', 'Grid Data\n(Load, Price)')
        c.node('time_data', 'Time Data\n(Hour, Day)')
    
    with dot.subgraph(name='cluster_1') as c:
        c.attr(label='RL Agent')
        c.node('state', 'State\nProcessor')
        c.node('policy', 'Policy\nNetwork')
        c.node('value', 'Value\nNetwork')
    
    with dot.subgraph(name='cluster_2') as c:
        c.attr(label='Environment')
        c.node('action', 'Action\nExecution')
        c.node('reward', 'Reward\nCalculation')
        c.node('next_state', 'Next State\nGeneration')
    
    with dot.subgraph(name='cluster_3') as c:
        c.attr(label='Output')
        c.node('schedule', 'Charging\nSchedule')
        c.node('metrics', 'Performance\nMetrics')
    
    # Add edges
    dot.edge('ev_data', 'state')
    dot.edge('grid_data', 'state')
    dot.edge('time_data', 'state')
    dot.edge('state', 'policy')
    dot.edge('state', 'value')
    dot.edge('policy', 'action')
    dot.edge('action', 'reward')
    dot.edge('reward', 'next_state')
    dot.edge('next_state', 'state')
    dot.edge('action', 'schedule')
    dot.edge('reward', 'metrics')
    
    # Save the diagram
    os.makedirs('docs/images', exist_ok=True)
    dot.render('docs/images/system_architecture', format='png', cleanup=True)

def create_training_flow():
    """Create a training flow diagram."""
    dot = Digraph(comment='Training Flow')
    dot.attr(rankdir='TB')
    
    # Add nodes
    dot.node('init', 'Initialize\nEnvironment')
    dot.node('collect', 'Collect\nExperience')
    dot.node('process', 'Process\nBatch')
    dot.node('update', 'Update\nPolicy')
    dot.node('eval', 'Evaluate\nPerformance')
    
    # Add edges
    dot.edge('init', 'collect')
    dot.edge('collect', 'process')
    dot.edge('process', 'update')
    dot.edge('update', 'eval')
    dot.edge('eval', 'collect')
    
    # Save the diagram
    dot.render('docs/images/training_flow', format='png', cleanup=True)

if __name__ == "__main__":
    create_system_architecture()
    create_training_flow() 