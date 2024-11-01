import os
import logging
import paho.mqtt.client as mqtt
import base64

# Configuration
broker = '192.168.1.79'
port = 1883
topic = 'image'
received_folder = 'received_images'  # change the directory/folder

# Set up logging
logging.basicConfig(filename='mqtt_image_receiver.log', level=logging.INFO,
                    format='%(asctime)s %(levelname)s:%(message)s')

# MQTT callbacks
def on_connect(client, userdata, flags, rc):
    logging.info(f"Connected with result code {rc}")
    print(f"Connected with result code {rc}")
    client.subscribe(topic)

def on_message(client, userdata, msg):
    try:
        # Decode image
        image_data = base64.b64decode(msg.payload)
        
        # Save image to file with fixed name
        os.makedirs(received_folder, exist_ok=True)
        image_path = os.path.join(received_folder, 'img.jpg')
        with open(image_path, 'wb') as file:
            file.write(image_data)
        
        logging.info(f"Image saved to {image_path}")
    except Exception as e:
        logging.error(f"Failed to save image: {e}")

# Function to start the MQTT receiver
def start_mqtt_receiver():
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message

    client.connect(broker, port, 60)
    client.loop_start()

    try:
        while True:
            pass  # Keep the script running to receive messages
    except KeyboardInterrupt:
        logging.info("Keyboard interrupt detected. Stopping the script.")
    except Exception as e:
        logging.error(f"Unexpected error occurred: {e}")
    finally:
        client.loop_stop()
        client.disconnect()

if __name__ == "__main__":
    start_mqtt_receiver()