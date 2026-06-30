import mujoco
import mujoco.viewer

# A simple XML string defining a scene with a plane and a falling box
xml = """
<mujoco>
  <compiler angle="degree"/>
  <option gravity="0 0 -9.81"/>
  
  <worldbody>
    <light pos="0 0 1.5" dir="0 0 -1" directional="true"/>
    <geom type="plane" size="1 1 0.1" rgba=".9 .9 .9 1"/>
    
    <body name="box" pos="0 0 1">
      <freejoint/>
      <geom type="box" size=".1 .1 .1" rgba="0 0 1 1" mass="1"/>
    </body>
  </worldbody>
</mujoco>
"""

def main():
    print("Loading MuJoCo model...")
    # Load the model from the XML string
    model = mujoco.MjModel.from_xml_string(xml)
    
    # Create the data object which holds the simulation state
    data = mujoco.MjData(model)
    
    print("Starting the viewer...")
    # Launch the interactive viewer
    mujoco.viewer.launch(model, data)

if __name__ == "__main__":
    main()
