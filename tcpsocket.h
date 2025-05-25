#pragma once
#ifndef _TCPSOCKET_H
#define _TCPCOCKET_H
#include<WinSock2.h>
#pragma comment(lib,"ws2_32.lib")
#include<iostream>
#define err(errMsg) cout << errMsg<< " failed code " << WSAGetLastError() << endl
using namespace std;
bool init_socket();
bool close_socket();
SOCKET creatservesocket();
SOCKET creatclientsocket();
#endif