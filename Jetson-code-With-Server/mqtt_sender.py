import paho.mqtt.client as mqtt

# Configuration
broker = '192.168.1.79'
port = 1883
publish_topic = 'jetson_posture_info'  # Topic to publish posture information

# Callback for when the client connects to the broker
def on_connect(client, userdata, flags, rc):
    print(f"Connected with result code {rc}")

# Function to send posture message
def send_posture_message(position):
    client = mqtt.Client()
    client.on_connect = on_connect

    client.connect(broker, port, 60)

    if position == 1:
        message = "1"  # Hunchback
    elif position == -1:
        message = "-1"  # Reclined
    else:
        message = "0"  # Straight

    client.publish(publish_topic, message)
    print(f"Sent posture message: {message}")

    client.disconnect()

# Example usage
if __name__ == "__main__":
    # Replace this with the actual position value
    #example_position = 1  # For testing
    send_posture_message()