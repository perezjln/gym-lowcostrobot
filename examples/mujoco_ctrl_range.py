import mujoco

if __name__ == "__main__":
    # Load the model from the XML file
    model = mujoco.MjModel.from_xml_path("assets/scene_one_cube.xml")

    # Extract control range for each actuator
    control_ranges = model.actuator_ctrlrange
    print(type(control_ranges))

    # Print the control ranges
    for i, (lower, upper) in enumerate(control_ranges):
        print(f"Actuator {i} control range: lower = {lower}, upper = {upper}")
