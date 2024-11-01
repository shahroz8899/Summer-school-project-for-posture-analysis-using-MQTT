# local_server.py
import paho.mqtt.client as mqtt
import os
import time
import logging
import base64

# Configuration
broker = 'localhost'  # Local server's own broker
port = 1883
image_topic = 'image'
posture_topic = 'posture_info'
jetson_image_topic = 'jetson_image_info'
jetson_posture_topic = 'jetson_posture_info'
image_save_path = './received_images'

# Set up logging
logging.basicConfig(level=logging.INFO)

def on_connect(client, userdata, flags, rc):
    logging.info(f"Connected with result code {rc}")
    client.subscribe(image_topic)
    client.subscribe(jetson_posture_topic)

def on_message(client, userdata, msg):
    if msg.topic == image_topic:
        # Decode and save the received image
        image_data = base64.b64decode(msg.payload)
        image_path = os.path.join(image_save_path, f'image_{int(time.time())}.jpg')
        os.makedirs(image_save_path, exist_ok=True)
        with open(image_path, 'wb') as f:
            f.write(image_data)
        logging.info(f"Received image and saved to {image_path}")
        
        # Forward the image to Jetson device
        client.publish(jetson_image_topic, msg.payload)
        logging.info("Image forwarded to Jetson device")
        
    elif msg.topic == jetson_posture_topic:
        # Forward the posture message to the first Raspberry Pi
        client.publish(posture_topic, msg.payload)
        logging.info("Posture message forwarded to first Raspberry Pi")

def main():
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    
    logging.info(f"Connecting to broker {broker}:{port}")
    client.connect(broker, port, 60)
    client.loop_forever()

if __name__ == "__main__":
    main()