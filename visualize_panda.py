import os
import mujoco
import mujoco.viewer

def main():
    # Construct the path to the scene.xml inside the mujoco_menagerie submodule
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "mujoco_menagerie", "franka_emika_panda", "scene.xml")
    
    if not os.path.exists(model_path):
        print(f"Error: Model path does not exist: {model_path}")
        return

    print(f"Loading MuJoCo model from: {model_path}")
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    print("Launching MuJoCo viewer... Use the side panel to control actuators/joints.")
    mujoco.viewer.launch(model, data)

if __name__ == "__main__":
    main()
