from vault import *
import socket
from server import MAX_DATA_SIZE
from threading import Thread

SERVER_IP = '127.0.0.1'
SERVER_PORT = 40300


class Client:
    def __init__(self, mpw, pin, T=20):
        self.user_name = 'admin'
        self.mpw = mpw
        self.pin = pin
        self.T = T
        self.vault = Vault(mpw, pin, T=T)

        self.s = socket.socket()

    def run(self, ip=SERVER_IP, port=SERVER_PORT):
        self.s.connect((ip, port))
        data = json.loads(self.s.recv(1024).decode())
        if data['op_type'] == 'Info' and data['msg'] == 'connect_success':
            print('connect to server!')
        else:
            print('connect fail')
            self.s.close()

    def msg_send(self, kind):
        resp = {'op_type': 'info', 'msg': kind}
        self.s.send(json.dumps(resp).encode())
        
    def byte_send(self, data):
        msg = json.loads(self.s.recv(1024).decode())
        if msg['op_type'] != 'recv_start':
            print('server not ready for recv!')
            return

        self.s.send(data)
        msg = json.loads(self.s.recv(1024).decode())
        if msg['op_type'] != 'recv_done':
            print('something wrong in data send!')
            return

    def file_send(self, file_path):
        with open(file_path, 'rb') as rf:
            req = {'op_type:': 'backup_file', 'user_name': self.user_name, 'file_name': file_path, 'new_file': True}
            data = rf.read(MAX_DATA_SIZE)
            while data:
                req['file_size'] = len(data)
                self.s.send(json.dumps(req).encode())
                self.byte_send(data)

                data = rf.read(MAX_DATA_SIZE)
                req['new_file'] = False

            print('file send success!')

# =============================================================================

    def add_pw(self, pw, dm):
        if self.vault.exist_dm(dm):
            index = self.vault.index[get_hash(dm)]

            self.vault.add_pw(pw, dm)
            data = bytes()
            for x in range(self.T):
                with open('data/vault_data/vault_{}'.format(x), 'rb') as rf:
                    cipher = rf.read()
                    data += cipher[index * MAX_PW_LENGTH * 4: (index+1) * MAX_PW_LENGTH * 4]

            req = {'op_type': 'modify_cipher', 'user_name': self.user_name, 'T': self.T,
                   'index': index, 'file_size': len(data)}
            self.s.send(json.dumps(req).encode())
            self.byte_send(data)

        else:
            self.vault.add_pw(pw, dm)
            print('new domain, prepare to backup vault file!')
            self.vault_backup()

        print('add password success!')

    def vault_backup(self):
        for x in range(self.T):
            self.file_send('data/vault_data/vault_{}'.format(x))
        print('backup vault file success')
