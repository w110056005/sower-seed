import zmq
import subprocess


print("Running Sower seed in background...")

def execute_python_file(file_path):
    try:
        # Execute the Python file as a separate process
        subprocess.run(['python', file_path], check=True)
    except subprocess.CalledProcessError as e:
        # Handle any errors that occur during the execution
        print(f"Error executing {file_path}: {e}")

# Create a ZeroMQ context
context = zmq.Context()
# Create a subscriber socket
socket = context.socket(zmq.SUB)
# Set the subscription filter (empty string means subscribe to all messages)
socket.setsockopt_string(zmq.SUBSCRIBE, '')
# Connect to the publisher's address
socket.connect('tcp://sower_platform_container:5555')

while True:
    print('Listening')
    message = socket.recv_string()
    print(f'Received message: {message}')
    
    if(message == "Start"):
        execute_python_file("client.py")
