"""
Simplified SAR Agent Architecture Diagram
Focus on agent decision-making and learning loop
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
import matplotlib.lines as mlines

# Create figure
fig, ax = plt.subplots(1, 1, figsize=(16, 12))
ax.set_xlim(0, 16)
ax.set_ylim(0, 12)
ax.axis('off')

# Color scheme
colors = {
    'input': '#667eea',
    'reasoning': "#0aa37a",
    'execution': '#ffd166',
    'feedback': '#ef476f',
    'learning': '#8338ec'
}

# Title
ax.text(8, 11.5, 'SAR Agent Architecture', 
        ha='center', va='top', fontsize=32, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#667eea', linewidth=3))

# ========== 1. INPUT LAYER ==========
input_y = 9.2
ax.add_patch(FancyBboxPatch((1, input_y), 4, 1.2, 
                            boxstyle="round,pad=0.1", 
                            facecolor=colors['input'], alpha=0.9, 
                            edgecolor='#333', linewidth=2))
ax.text(3, input_y + 0.9, 'INPUT', ha='center', fontsize=14, fontweight='bold', color='white')
ax.text(3, input_y + 0.5, 'Natural Language', ha='center', fontsize=11, color='white')
ax.text(3, input_y + 0.15, '"Search 4 urban blocks"', ha='center', fontsize=9, 
        color='white', style='italic')

# ========== 2. LLM REASONING ==========
llm_y = 7.0
ax.add_patch(FancyBboxPatch((0.5, llm_y), 5, 1.5, 
                            boxstyle="round,pad=0.1", 
                            facecolor=colors['reasoning'], alpha=0.9, 
                            edgecolor='#333', linewidth=2))
ax.text(3, llm_y + 1.3, 'REASONING', ha='center', fontsize=14, fontweight='bold', color='white')
ax.text(3, llm_y + 0.95, 'LLM Planner', ha='center', fontsize=12, color='white')
ax.text(3, llm_y + 0.75, 'Llama 3.2 1B', ha='center', fontsize=10, color='white', style='italic')
ax.text(3, llm_y + 0.25, '• Parse intent\n• Call tools\n• Generate plan', 
        ha='center', fontsize=9, color='white')

# Tool box
ax.add_patch(FancyBboxPatch((6, llm_y + 0.35), 2.5, 1.1, 
                            boxstyle="round,pad=0.05", 
                            facecolor='white', edgecolor=colors['reasoning'], linewidth=2))
ax.text(7.25, llm_y + 1.2, 'Tools', ha='center', fontsize=11, fontweight='bold', color=colors['reasoning'])
ax.text(7.25, llm_y + 0.95, 'Grid Search', ha='center', fontsize=8)
ax.text(7.25, llm_y + 0.75, 'Spiral Search', ha='center', fontsize=8)
ax.text(7.25, llm_y + 0.55, 'Divide Area', ha='center', fontsize=8)

# Arrow from input to LLM
arrow = FancyArrowPatch((3, input_y), (3, llm_y + 1.5), 
                       arrowstyle='->', mutation_scale=40, 
                       linewidth=3, color='#333')
ax.add_patch(arrow)

# Arrow LLM to tools
arrow = FancyArrowPatch((5.56, llm_y + 0.95), (6, llm_y + 0.95), 
                       arrowstyle='<->', mutation_scale=20, 
                       linewidth=2, color=colors['reasoning'])
ax.add_patch(arrow)

# ========== 3. MDP EVALUATION ==========
mdp_y = llm_y
ax.add_patch(FancyBboxPatch((9, mdp_y), 6, 1.5, 
                            boxstyle="round,pad=0.1", 
                            facecolor=colors['feedback'], alpha=0.9, 
                            edgecolor='#333', linewidth=2))
ax.text(12, mdp_y + 1.2, 'EVALUATION', ha='center', fontsize=14, fontweight='bold', color='white')
ax.text(12, mdp_y + 0.85, 'MDP Scorer', ha='center', fontsize=12, color='white')

# Two evaluation types
ax.add_patch(FancyBboxPatch((9.5, mdp_y + 0.1), 2, 0.6, 
                            boxstyle="round,pad=0.05", 
                            facecolor='white', edgecolor=colors['feedback'], linewidth=1))
ax.text(10.5, mdp_y + 0.5, 'Plan Metrics', ha='center', fontsize=9, fontweight='bold')
ax.text(10.5, mdp_y + 0.2, 'Coverage, Efficiency', ha='center', fontsize=7)

ax.add_patch(FancyBboxPatch((12, mdp_y + 0.1), 2.5, 0.6, 
                            boxstyle="round,pad=0.05", 
                            facecolor='white', edgecolor=colors['feedback'], linewidth=1))
ax.text(13.25, mdp_y + 0.5, 'State Transitions', ha='center', fontsize=9, fontweight='bold')
ax.text(13.25, mdp_y + 0.2, 'Diminishing Returns', ha='center', fontsize=7)

# Arrow from LLM to MDP
arrow = FancyArrowPatch((5.5, llm_y + 0.1), (9, mdp_y + 0.1), 
                       arrowstyle='->', mutation_scale=40, 
                       linewidth=3, color='#333')
ax.add_patch(arrow)
ax.text(7.25, llm_y - 0.1, 'Mission Plan', ha='center', fontsize=10, fontweight='bold')

# ========== 4. MISSION PLAN OUTPUT ==========
plan_y = 5.2
ax.add_patch(FancyBboxPatch((1, plan_y), 4, 1.0, 
                            boxstyle="round,pad=0.1", 
                            facecolor=colors['execution'], alpha=0.9, 
                            edgecolor='#333', linewidth=2))
ax.text(3, plan_y + 0.75, 'MISSION PLAN', ha='center', fontsize=14, fontweight='bold', color='#333')
ax.text(3, plan_y + 0.4, 'High-level Actions', ha='center', fontsize=11, color='#333')
ax.text(3, plan_y + 0.1, 'Waypoints, search patterns', ha='center', fontsize=8, 
        color='#333', style='italic')

# Arrow from LLM to plan
arrow = FancyArrowPatch((3, llm_y), (3, plan_y + 1.0), 
                       arrowstyle='->', mutation_scale=40, 
                       linewidth=3, color='#333')
ax.add_patch(arrow)

# ========== 5. RL CONTROLLER ==========
rl_y = 3.0
ax.add_patch(FancyBboxPatch((0.5, rl_y), 5, 1.4, 
                            boxstyle="round,pad=0.1", 
                            facecolor='#118ab2', alpha=0.9, 
                            edgecolor='#333', linewidth=2))
ax.text(3, rl_y + 1.15, 'RL CONTROLLER', ha='center', fontsize=14, fontweight='bold', color='white')
ax.text(3, rl_y + 0.85, 'Low-level Control', ha='center', fontsize=12, color='white')
ax.text(3, rl_y + 0.5, '• Waypoint following', ha='center', fontsize=9, color='white')
ax.text(3, rl_y + 0.3, '• Obstacle avoidance', ha='center', fontsize=9, color='white')
ax.text(3, rl_y + 0.1, '• Motor commands', ha='center', fontsize=9, color='white')

# Sensors box
ax.add_patch(FancyBboxPatch((6, rl_y + 0.09), 2.5, 1.3, 
                            boxstyle="round,pad=0.05", 
                            facecolor='white', edgecolor='#118ab2', linewidth=2))
ax.text(7.25, rl_y + 1.1, 'Sensors', ha='center', fontsize=11, fontweight='bold', color='#118ab2')
ax.text(7.25, rl_y + 0.9, 'IMU', ha='center', fontsize=8)
ax.text(7.25, rl_y + 0.7, 'GPS', ha='center', fontsize=8)
ax.text(7.25, rl_y + 0.5, 'Altitude', ha='center', fontsize=8)
ax.text(7.25, rl_y + 0.3, 'Camera', ha='center', fontsize=8)

# Arrow from plan to RL
arrow = FancyArrowPatch((3, plan_y), (3, rl_y + 1.4), 
                       arrowstyle='->', mutation_scale=40, 
                       linewidth=3, color='#333')
ax.add_patch(arrow)
ax.text(3.7, plan_y - 0.3, 'Waypoints', ha='center', fontsize=9, fontweight='bold')

# Arrow sensors to RL
arrow = FancyArrowPatch((6, rl_y + 0.7), (5.5, rl_y + 0.7), 
                       arrowstyle='->', mutation_scale=20, 
                       linewidth=2, color='#118ab2')
ax.add_patch(arrow)

# Motor output arrow
arrow = FancyArrowPatch((3, rl_y), (3, rl_y - 0.5), 
                       arrowstyle='->', mutation_scale=40, 
                       linewidth=3, color='#333')
ax.add_patch(arrow)
ax.text(3, rl_y - 0.7, 'Motor Commands', ha='center', fontsize=10, fontweight='bold', color='#333')
ax.text(3, rl_y - 0.9, '(thrust, pitch, roll, yaw)', ha='center', fontsize=8, style='italic', color='#666')

# ========== 6. LEARNING LOOP ==========
learn_y = 0.5
ax.add_patch(FancyBboxPatch((0.5, learn_y), 14.5, 1.2, 
                            boxstyle="round,pad=0.1", 
                            facecolor=colors['learning'], alpha=0.9, 
                            edgecolor='#333', linewidth=3))
ax.text(7.75, learn_y + 0.95, 'LEARNING LOOP', ha='center', fontsize=14, fontweight='bold', color='white')

# Training steps
steps = [
    ('1. Collect', 1.5, learn_y + 0.35, 'Generate diverse\nmissions'),
    ('2. Score', 4.5, learn_y + 0.35, 'MDP evaluation\n(0.0 - 1.0)'),
    ('3. Dataset', 7.5, learn_y + 0.35, 'Mission + Plan\n+ Score'),
    ('4. Fine-tune', 10.5, learn_y + 0.35, 'LoRA training\n(~1% params)'),
    ('5. Deploy', 13, learn_y + 0.35, 'Updated model\nback to agent'),
]

for i, (label, x, y, desc) in enumerate(steps):
    # Circle with number
    circle_color = 'white'
    ax.add_patch(Circle((x, y + 0.35), 0.2, facecolor=circle_color, edgecolor='#333', linewidth=2))
    ax.text(x, y + 0.35, label.split('.')[0], ha='center', va='center', 
           fontsize=10, fontweight='bold', color=colors['learning'])
    
    # Description
    ax.text(x, y, label.split('.')[1].strip(), ha='center', va='top', 
           fontsize=8, fontweight='bold', color='white')
    ax.text(x, y - 0.15, desc, ha='center', va='top', 
           fontsize=6, color='white')
    
    # Arrow to next step
    if i < len(steps) - 1:
        next_x = steps[i+1][1]
        arrow = FancyArrowPatch((x + 0.25, y + 0.35), (next_x - 0.25, y + 0.35), 
                               arrowstyle='->', mutation_scale=15, 
                               linewidth=2, color='white')
        ax.add_patch(arrow)

# Arrow from MDP to learning
# arrow = FancyArrowPatch((12, mdp_y), (5.4, learn_y + 1.3), 
#                        arrowstyle='->', mutation_scale=30, 
#                        linewidth=2, color=colors['learning'], linestyle='dashed')
# ax.add_patch(arrow)
# ax.text(11.5, mdp_y - 0.45, 'Scores', ha='center', fontsize=9, color=colors['learning'], fontweight='bold')

# Arrow from learning back to LLM
# arrow = FancyArrowPatch((13, learn_y + 1.2), (4, llm_y), 
#                        arrowstyle='->', mutation_scale=30, 
#                        linewidth=2, color=colors['learning'], linestyle='dashed')
# ax.add_patch(arrow)
# ax.text(6.7, llm_y - 0.45, 'Improved Model', ha='center', fontsize=9, color=colors['learning'], fontweight='bold')

# ========== KEY INSIGHT BOX ==========
insight_y = rl_y
ax.add_patch(FancyBboxPatch((9.5, insight_y), 5.5, 1.4, 
                            boxstyle="round,pad=0.1", 
                            facecolor='#f8f9fa', alpha=0.95, 
                            edgecolor='#333', linewidth=2, linestyle='--'))
ax.text(12.25, insight_y + 0.95, 'KEY INSIGHT', ha='center', fontsize=10, fontweight='bold', color='#333')
ax.text(12.25, insight_y + 0.65, 'Agent learns optimal planning', ha='center', fontsize=9, color='#333')
ax.text(12.25, insight_y + 0.4, 'through MDP reward feedback', ha='center', fontsize=9, color='#333')
ax.text(12.25, insight_y + 0.1, 'No human labeling required!', ha='center', fontsize=8, 
       color=colors['learning'], fontweight='bold', style='italic')

plt.tight_layout()
plt.savefig('res/agent_architecture.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('res/agent_architecture.pdf', bbox_inches='tight', facecolor='white')

print("\n" + "="*60)
print("✅ Simplified agent architecture generated!")
print("="*60)
print("\n📁 Files created:")
print("   • agent_architecture.png (High-res PNG)")
print("   • agent_architecture.pdf (Vector PDF)")
print("\n🎯 Shows:")
print("   • Input → LLM Reasoning → Mission Plan")
print("   • Mission Plan → RL Controller → Motor Commands")
print("   • MDP evaluation feedback")
print("   • Learning loop (collect → score → train → deploy)")
print("   • Sensor feedback to RL controller")
print("\n✨ Clean, focused on agent decision-making!")
print("="*60 + "\n")
