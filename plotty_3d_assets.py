import os
import numpy as np
import plotly.graph_objects as go

def generate_animated_3d_plot(data_path="simulated_data/flight_telemetry_gravity_turn_3d_displacement.npz", step_interval=30):
    output_dir = "simulated_data/3d_assets"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load multi-dimensional physics arrays from disk
    try:
        data = np.load(data_path)
    except FileNotFoundError:
        print(f"Error: The simulation file '{data_path}' was not found. Please run your simulation.py script first.")
        return

    time = data["time"]
    disp = data["displacement"]  # Shape (N, 3) -> [X, Y, Z]
    vel = data["velocity"]        # Shape (N, 3) -> [Vx, Vy, Vz]
    
    velocity_mag = np.linalg.norm(vel, axis=1)

    # Slice the data to optimize browser rendering performance
    indices = list(range(0, len(time), step_interval))
    if (len(time) - 1) not in indices:
        indices.append(len(time) - 1)

    t_sliced = time[indices]
    disp_sliced = disp[indices]
    vel_sliced = velocity_mag[indices]

    # 2. Setup the Initial Base State (Frame 0)
    fig = go.Figure(
        data=[
            # Trace 0: The growing trajectory line trace
            go.Scatter3d(
                x=[disp_sliced[0, 0]],
                y=[disp_sliced[0, 1]],
                z=[disp_sliced[0, 2]],
                mode='lines',
                name='Flight Path',
                line=dict(color='#00F5FF', width=5),
                hovertemplate=(
                    "<b>Telemetry Snapshot</b><br>" +
                    "Time: %{customdata:.2f} s<br>" +
                    "Downrange X: %{x:.1f} m<br>" +
                    "Crossrange Y: %{y:.1f} m<br>" +
                    "Altitude Z: %{z:.1f} m<br>" +
                    "Velocity: %{customdata:.2f} m/s" +
                    "<extra></extra>"
                ),
                customdata=np.stack((t_sliced, vel_sliced), axis=-1)
            ),
            # Trace 1: The active rocket marker dot
            go.Scatter3d(
                x=[disp_sliced[0, 0]],
                y=[disp_sliced[0, 1]],
                z=[disp_sliced[0, 2]],
                mode='markers',
                name='Rocket Core',
                # FIXED: Reduced marker size down to 3 for a sleeker profile layout
                marker=dict(color='#FFFFFF', size=3)
            )
        ]
    )

    # 3. Compile Sequential Chronological Animation Frames
    frames = []
    slider_steps = []

    for i, t_val in enumerate(t_sliced):
        frame_name = f"Frame_{i}"
        
        frames.append(
            go.Frame(
                data=[
                    go.Scatter3d(
                        x=disp_sliced[:i+1, 0],
                        y=disp_sliced[:i+1, 1],
                        z=disp_sliced[:i+1, 2]
                    ),
                    go.Scatter3d(
                        x=[disp_sliced[i, 0]],
                        y=[disp_sliced[i, 1]],
                        z=[disp_sliced[i, 2]]
                    )
                ],
                name=frame_name
            )
        )
        
        slider_steps.append(
            dict(
                method="animate",
                label=f"{t_val:.1f}s",
                args=[
                    [frame_name],
                    dict(
                        mode="immediate",
                        transition=dict(duration=0),
                        frame=dict(duration=0, redraw=True)
                    )
                ]
            )
        )

    fig.frames = frames

    # 4. Implement Media Controls (Play Button, Pause Button, and Timeline Slider)
    fig.update_layout(
        title=dict(
            text="INTERACTIVE 3D AEROSPACE FLIGHT PROFILE WITH TIME CONTROL",
            x=0.5, y=0.95,
            font=dict(color='white', size=16, family="Arial")
        ),
        paper_bgcolor='black',
        plot_bgcolor='black',
        showlegend=False,
        scene=dict(
            xaxis=dict(
                title='Downrange X (m)', backgroundcolor='rgb(10, 10, 10)',
                gridcolor='rgba(128, 128, 128, 0.15)', showbackground=True,
                zerolinecolor='gray', tickfont=dict(color='gray'),
                range=[np.min(disp[:, 0]), np.max(disp[:, 0])]
            ),
            yaxis=dict(
                title='Crossrange Y (m)', backgroundcolor='rgb(10, 10, 10)',
                gridcolor='rgba(128, 128, 128, 0.15)', showbackground=True,
                zerolinecolor='gray', tickfont=dict(color='gray'),
                range=[np.min(disp[:, 1]), np.max(disp[:, 1])]
            ),
            zaxis=dict(
                title='Altitude Z (m)', backgroundcolor='rgb(20, 20, 20)',
                gridcolor='rgba(128, 128, 128, 0.15)', showbackground=True,
                zerolinecolor='gray', tickfont=dict(color='gray'),
                range=[np.min(disp[:, 2]), np.max(disp[:, 2])]
            ),
            camera=dict(eye=dict(x=1.5, y=-1.5, z=1.2))
        ),
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                x=0.05, y=0.05,
                xanchor="left", yanchor="top",
                direction="left",
                pad=dict(t=0, r=10),
                font=dict(color="white"),
                bgcolor="rgb(30,30,30)",
                buttons=[
                    dict(
                        label="▶ Play",
                        method="animate",
                        args=[
                            None,
                            dict(
                                frame=dict(duration=30, redraw=True),
                                fromcurrent=True,
                                transition=dict(duration=0),
                                mode="immediate"
                            )
                        ]
                    ),
                    dict(
                        label="❚❚ Pause",
                        method="animate",
                        args=[
                            [None],
                            dict(
                                frame=dict(duration=0, redraw=False),
                                mode="immediate"
                            )
                        ]
                    )
                ]
            )
        ],
        sliders=[
            dict(
                active=0,
                steps=slider_steps,
                x=0.15, y=0.05,
                currentvalue=dict(
                    font=dict(size=14, color="#00F5FF"),
                    prefix="Current Mission Time: ",
                    visible=True,
                    xanchor="right"
                ),
                pad=dict(t=0, b=10),
                len=0.8,
                font=dict(color="gray")
            )
        ],
        margin=dict(r=0, l=0, b=40, t=60)
    )

    # 5. Export as an isolated web asset container file
    html_output_path = os.path.join(output_dir, "flight_trajectory_animated_3d.html")
    print(f"\nCompiling dynamic interactive model... Saving to: '{html_output_path}'")
    
    fig.write_html(html_output_path, auto_open=True)
    print("SUCCESS: 3D interactive layout asset generated. Opening in web browser...")

if __name__ == "__main__":
    generate_animated_3d_plot(step_interval=20)
