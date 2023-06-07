import socket

class MySocket():
    def __init__(self):
        
        # Создание TCP сокета
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        # Привязка сокета к адресу и порту
        self.sock.bind(('localhost', 9090))

        # Ожидание подключения клиента
        self.sock.listen()
        # self.client_socket, self.client_address = self.sock.accept()
        # self.client_socket.setblocking(False)
        # print('Client connected: ', self.client_address)

    def send_data(self, x, y):
        client_socket, client_address = self.sock.accept()
        print('Client connected: ', client_address)
        client_socket.setblocking(False)
        
        # Отправка координат клиенту
        data = f"{x},{y}".encode()
        self.client_socket.sendall(data)

        # Закрытие соединения с клиентом
        self.client_socket.close()