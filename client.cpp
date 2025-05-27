#include<iostream>
#include"cmath"
#include"fstream"
#include<istream>
#include<opencv2/opencv.hpp>
#include"tcpsocket.h"
#include<thread>
#include<vector>
#include <mutex>
static mutex mtx;
using namespace cv;
using namespace std;
template<typename T> class Juzhen;
template<typename T>
class Juzhen
{
public:
	int hang;
	int lie;
	T** locate;
	Juzhen(int hang, int lie)
	{
		this->hang = hang;
		this->lie = lie;
		this->locate = new T * [hang];
		for (int i = 0; i < hang; i++)
		{
			locate[i] = new T[lie]();
		}
	}
	Juzhen()
	{

	}
	~Juzhen()
	{
		if (locate != NULL)
		{
			for (int i = 0; i < hang; i++)
			{
				delete[] locate[i];
			}
			delete[] locate;
			locate = NULL;
		}
	}
	Juzhen operator+(const Juzhen<T>& a)const
	{
		if (this->hang != a.hang || this->lie != a.lie)
		{
			cout << "输入有误，不能加法" << endl;
			return Juzhen<T>(0, 0);
		}
		else
		{
			Juzhen<T> result(hang, lie);
			for (int i = 0; i < this->hang; i++)
			{
				for (int j = 0; j < this->lie; j++)
				{
					result.locate[i][j] = a.locate[i][j] + this->locate[i][j];
				}
			}
			return result;
		}
	}


	Juzhen operator*(const Juzhen<T>& a)const
	{

		if (lie != a.hang)
		{
			cerr << "矩阵维度不匹配，无法相乘" << endl;
			return Juzhen<T>(0, 0);
		}
		Juzhen<T> result(hang, a.lie);
		thread_fun(*this, a, result);
		return result;
	}

	Juzhen(const Juzhen<T>& a) : hang(a.hang), lie(a.lie)
	{
		locate = new T * [hang];
		for (int i = 0; i < hang; i++)
		{
			locate[i] = new T[lie];
			for (int j = 0; j < lie; j++)
			{
				locate[i][j] = a.locate[i][j];
			}
		}
	}
	Juzhen<T>& operator=(const Juzhen<T>& a)
	{
		if (this == &a) return *this;
		for (int i = 0; i < hang; i++)
		{
			delete[] locate[i];
		}
		delete[] locate;
		hang = a.hang;
		lie = a.lie;
		locate = new T * [hang];
		for (int i = 0; i < hang; i++)
		{
			locate[i] = new T[lie];
			for (int j = 0; j < lie; j++)
			{
				locate[i][j] = a.locate[i][j];
			}
		}
		return *this;
	}
	void bianli()const
	{
		for (int i = 0; i < this->hang; i++)
		{
			for (int j = 0; j < this->lie; j++)
			{
				cout << this->locate[i][j] << " ";
			}
			cout << endl;
		}
	}
};
template<typename T>
void thread_task(const Juzhen<T>& a, const Juzhen<T>& b, Juzhen<T>& c, int start_row, int end_row)
{

	for (int i = start_row; i < end_row; i++)
	{
		for (int k = 0; k < b.lie; k++)
		{
			for (int j = 0; j < a.lie; ++j)
			{
				lock_guard<mutex> lock(mtx);
				c.locate[i][k] += a.locate[i][j] * b.locate[j][k];
			}
		}
	}
}
template<typename T>
void thread_fun(const Juzhen<T>& a, const Juzhen<T>& b, Juzhen<T>& c)
{
	const int THREAD_COUNT = max(1, thread::hardware_concurrency() / 2);
	vector<thread> threads;
	int rows_per_thread = a.hang / THREAD_COUNT;
	for (int t = 0; t < THREAD_COUNT; t++)
	{
		int start = t * rows_per_thread;
		int end = (t == THREAD_COUNT-1) ? a.hang : start + rows_per_thread;
		threads.emplace_back([&a, &b, &c, start, end] {thread_task(a, b, c, start, end); });
	}
	for (auto& t : threads)
	{
		t.join();
	}
}
template <typename T>
Juzhen<T> buildjuzhen(const T* data, int hang, int lie)
{
	Juzhen<T> m(hang, lie);
	for (int i = 0; i < hang; i++)
	{
		for (int j = 0; j < lie; j++)
		{
			m.locate[i][j] = data[i * lie + j];
		}
	}
	return m;
}
template<typename T>
T* sendjuzhen(const Juzhen<T>& m) 
{
	T* data = new T[m.hang * m.lie];
	for (int i = 0; i < m.hang; i++) 
	{
		for (int j = 0; j < m.lie; j++) 
		{
			int index = i * m.lie + j;
			uint32_t net_val = htonl(*reinterpret_cast<uint32_t*>(&m.locate[i][j]));
			memcpy(&data[index], &net_val, sizeof(net_val));
		}
	}
	return data;
}
template<typename T>
Juzhen<T> readimg(string path)
{
	Mat image = imread(path, IMREAD_GRAYSCALE);
	if (image.empty())
	{
		cout << "读取图片失败" << endl;
		return Juzhen<T>(0, 0);
	} 
	Mat resize_image;
	resize(image, resize_image, Size(28, 28), 0, 0, INTER_AREA);

	Juzhen<T> m(1, 784);
	for (int i = 0; i < 28; i++)
	{
		for (int j = 0; j < 28; j++)
		{
			m.locate[0][28 * i + j] = resize_image.at<uchar>(i, j) / 255.0f;
		}
	}
	return m;
}
void cleanup(SOCKET serfd, SOCKET clifd) 
{
	closesocket(clifd);
	closesocket(serfd);
	WSACleanup();
}
int main()
{
	init_socket();
	SOCKET fd=creatclientsocket();
	Juzhen<float> convert = readimg<float>("8.png");
	cout << convert.hang << " " << convert.lie << endl;
	cout<< "---------start------------" << endl;
	
	float* data= sendjuzhen(convert);
	int arr[2] = {htonl(convert.hang), htonl(convert.lie)};
	int arr_size = sizeof(arr);
	int sent = 0;
	while (sent < arr_size) 
	{
		int ret = send(fd, (char*)arr + sent, arr_size - sent, 0);
		if (ret <= 0) 
		{
			err("send arr");
			break;
		}
		sent += ret;
	}
	int data_size = convert.hang * convert.lie * sizeof(float);
	sent = 0;
	while (sent < data_size)
	{
		int ret = send(fd, (char*)data + sent, data_size - sent, 0);
		if (ret <= 0)
		{
			err("send data");
			break;
		}
		sent += ret;
	}
	
	delete[] data;
	cout << "开始接收" << endl;
	int hang = 0, lie = 0;
	int recv_bytes = 0;
	while (recv_bytes < sizeof(hang))
	{
		int ret = recv(fd, reinterpret_cast<char*>(&hang) + recv_bytes, sizeof(hang) - recv_bytes, 0);
		if (ret <= 0)
		{
			err("receive " + __LINE__);
		}
		recv_bytes += ret;
	}
	hang = ntohl(hang);
	recv_bytes = 0;
	while (recv_bytes < sizeof(lie))
	{
		int ret = recv(fd, reinterpret_cast<char*>(&lie) + recv_bytes, sizeof(lie) - recv_bytes, 0);
		if (ret <= 0)
		{
			err("receive " + __LINE__);
		}
		recv_bytes += ret;
	}
	lie = ntohl(lie);
	cout << "接收矩阵维度: " << hang << "*" << lie << endl;
	vector<float> sentdata(hang * lie);
	int total_received = 0;
	while (total_received < sentdata.size()*sizeof(float))
	{
		int ret = recv(fd, reinterpret_cast<char*>(sentdata.data()) + total_received,sentdata.size() * sizeof(float) - total_received, 0);
		if (ret <= 0)
		{
			err("recv data");
		}
		total_received += ret;
	}
	for (int i = 0; i < sentdata.size(); i++) 
	{
		uint32_t net_val;
		memcpy(&net_val, &sentdata[i], sizeof(net_val));
		float local_val;
		*reinterpret_cast<uint32_t*>(&local_val) = ntohl(net_val);
		sentdata[i] = local_val;
	}
	Juzhen<float> final = buildjuzhen<float>(sentdata.data(), hang, lie);
	final.bianli();
	int max = 0;
	for (int i = 1; i < 10; i++)
	{
		if (final.locate[0][max] < final.locate[0][i])
		{
			max = i;
		}
	}
	cout << "最有可能的数字是"<< max << endl;
	cout << "---------end------------" << endl;
	getchar();
	close_socket();
}
