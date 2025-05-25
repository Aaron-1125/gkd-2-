#include"tcpsocket.h"
#define err(errMsg) cout << errMsg<< " failed code " << WSAGetLastError() << endl
#define PORT 8888
bool init_socket()
{
	WSADATA wasdata;
	if (0 != WSAStartup(MAKEWORD(2, 2), &wasdata))
	{
		err("WSAStartup");
		return false;
	}
	return true;
}

bool close_socket()
{
	if (0 != WSACleanup())
	{
		err("WSACleanup");
		return false;
	}
	return true;
}

SOCKET creatservesocket()
{
	SOCKET fd = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
	if (fd == INVALID_SOCKET) 
	{
		err("socket");
		return INVALID_SOCKET;
	}
	struct sockaddr_in addr;
	addr.sin_family = AF_INET;
	addr.sin_port = htons(PORT);
	addr.sin_addr.S_un.S_addr = htonl(INADDR_ANY);  
	if (bind(fd, (struct sockaddr*)&addr, sizeof(addr)) == SOCKET_ERROR)
	{
		err("bind");
		closesocket(fd);  
		return INVALID_SOCKET;
	}
	if (listen(fd, 10) == SOCKET_ERROR)
	{ 
		err("listen");
		closesocket(fd);
		return INVALID_SOCKET;
	}
	return fd; 
}

SOCKET creatclientsocket()
{
	SOCKET fd = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
	if (fd == INVALID_SOCKET)
	{
		err("socket");
		return INVALID_SOCKET;
	}
	struct sockaddr_in addr;
	addr.sin_family = AF_INET;
	addr.sin_port = htons(PORT);
	addr.sin_addr.S_un.S_addr = inet_addr("127.0.0.1");
	if (INVALID_SOCKET == connect(fd, (struct sockaddr*)&addr, sizeof(addr)))
	{
		err("connect");
		return false;
	}
	return fd;
	
}
