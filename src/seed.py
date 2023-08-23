from paho.mqtt import client as mqtt_client
import subprocess
import random


broker = 'broker.emqx.io'
port = 1883
topic = "Sower"
# Generate a Client ID with the subscribe prefix.
client_id = f'subscribe-{random.randint(0, 100)}'
# username = 'emqx'
# password = 'public'

def connect_mqtt() -> mqtt_client:
    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            print("Connected to MQTT Broker!")
        else:
            print("Failed to connect, return code %d\n", rc)

    client = mqtt_client.Client(client_id)
    # client.username_pw_set(username, password)
    client.on_connect = on_connect
    client.connect(broker, port)
    return client

def subscribe(client: mqtt_client):
    def on_message(client, userdata, msg):
        print('Receive msg: '+ msg.payload.decode())
        msg = msg.payload.decode().split(',')
        if(msg[0] == "Start"):
            execute_python_file("client.py", msg[1])

    client.subscribe(topic)
    client.on_message = on_message

def execute_python_file(file_path, port):
    try:
        # Execute the Python file as a separate process
        subprocess.run(['python', file_path, port], check=True)
    except subprocess.CalledProcessError as e:
        # Handle any errors that occur during the execution
        print(f"Error executing {file_path}: {e}")

while True:
    print("Running Sower client in background...")
    client = connect_mqtt()
    subscribe(client)
    client.loop_forever()