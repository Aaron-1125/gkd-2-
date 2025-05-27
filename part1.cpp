#include<iostream>
#include"cmath"
#include"fstream"
#include<istream>
#include<opencv2/opencv.hpp>
#include"tcpsocket.h"
#include<thread>
#include<vector>
#include <mutex>
#include"string"
using namespace cv;
using namespace std;
template<typename T> class Juzhen;
template<typename T> class model;
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
		this->locate = new T* [hang];
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
		locate = new T* [hang];
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
		locate = new T* [hang];
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
static mutex mtx;
template<typename T>
void thread_task(const Juzhen<T>& a,const Juzhen<T>& b,Juzhen<T>& c,int start_row,int end_row) 
{
	
	for (int i = start_row; i < end_row; i++) 
	{
		for (int k = 0; k <b.lie ; k++) 
		{
			for (int j = 0; j <a.lie; ++j) 
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
		threads.emplace_back([&a, &b, &c, start, end] {thread_task(a, b, c, start, end);});
	}
	for (auto& t : threads) 
	{
		t.join();
	}
}
template<typename T>
void readfile(const string& filename, int hang, int lie,Juzhen<T>& final)
{
	
	ifstream file(filename, ios::binary|ios::in);
	if (!file)
	{
		cout << "打开失败" << endl;
		file.close();
		final = Juzhen<T>(0, 0);
	}
	
	final.hang = hang;
	final.lie = lie;
	final.locate = new T * [hang];
	for (int i = 0; i < hang; i++)
	{
		final.locate[i] = new T[lie]();
	}
	for (int i = 0; i < hang; i++)
	{
		for (int j = 0; j < lie; j++)
		{
			T temp_num;
			file.read(reinterpret_cast<char*>(&temp_num), sizeof(T));
			final.locate[i][j] = temp_num;
		}
	}
	file.close();
}
template<typename T>
class model_father 
{
public:
	virtual ~model_father() = default;
	virtual Juzhen<T> forward(const Juzhen<T>& x) const = 0;
};
template<typename T>
class model : public model_father<T>
{
public:
	
	Juzhen<T> w1;
	Juzhen<T> b1;
	Juzhen<T> w2;
	Juzhen<T> b2;
	model(const string& path)
	{
		
		readfile<float>(path + "fc1.weight", 784, 500 , w1);
		
		 if (w1.hang == 0 || w1.lie == 0) 
		 {
			 cerr << "模型加载失败" << endl;
			 exit(EXIT_FAILURE);
		 }
		readfile<float>(path + "fc1.bias", 1, 500,b1);
		 if (b1.hang == 0 || b1.lie == 0)
		 {
			 cerr << "模型加载失败" << endl;
			 exit(EXIT_FAILURE);
		 }
		readfile<float>(path + "fc2.weight", 500, 10,w2);
		 if (w2.hang == 0 || w2.lie == 0) 
		 {
			 cerr << "模型加载失败" << endl;
			 exit(EXIT_FAILURE);
		 }
		readfile<float>(path + "fc2.bias", 1, 10,b2);
		 if (b2.hang == 0 || b2.lie == 0) 
		 {
			 cerr << "模型加载失败" << endl;
			 exit(EXIT_FAILURE);
		 }
		
	}
	Juzhen<T> forward(const Juzhen<T>& x) const
	{
		Juzhen<T> y = relu(x * w1 + b1);
		Juzhen<T> final = softmax(y * w2 + b2);
		if (final.hang == 0 || final.lie == 0)
		{
			cerr << "模型输出异常" << endl;
			return Juzhen<T>(1, 1);  
		}
		return final;
	}
};
template<typename T>
model_father<T>* create_model(const string& path)
{
	return new model<T>(path);
}
template <typename T>
void buildjuzhen(const T* data, int hang, int lie,Juzhen<T>& input )
{
	input.hang = hang;
	input.lie = lie;
	input.locate = new T* [hang];
	for (int i = 0; i < hang; i++) 
	{
		input.locate[i] = new T[lie]();
	}
	for (int i = 0; i < hang; i++)
	{
		for (int j = 0; j < lie; j++) 
		{
			input.locate[i][j] = data[i * lie + j];
		}
	}
	
}
template <typename T>
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
}template <typename T>
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
template <typename T>
Juzhen<T> relu(const Juzhen<T>& a)
{
	Juzhen<T> temp = a;
	for (int i = 0; i < a.hang; i++)
	{
		for (int j = 0; j < a.lie; j++)
		{
			if (temp.locate[i][j] < 0)
			{
				temp.locate[i][j] = 0;
			}
		}
	}
	return temp;
}
template <typename T>
Juzhen<T> softmax(const Juzhen<T>& a)
{
	Juzhen<T> temp = a;
	T sum = 0;
	if (a.hang == 1) 
	{
		for (int i = 0; i < a.lie; i++) 
		{
			sum += exp(a.locate[0][i]);
		}
		for (int i = 0; i < a.lie; i++) 
		{
			temp.locate[0][i] /= sum;
		}
		return temp;
	}
	if (a.lie == 1) 
	{
		for (int i = 0; i < a.hang; i++) 
		{
			sum += exp(a.locate[i][0]);
		}
		for (int i = 0; i < a.hang; i++)
		{
			temp.locate[i][0] /= sum;
		}
		return temp;
	}
	cout << "输入格式不符" << endl;
	return Juzhen<T>(0, 0);
}
void cleanup(SOCKET serfd, SOCKET clifd) 
{
	closesocket(clifd);
	closesocket(serfd);
	WSACleanup();
}
int main()
{

	if (!init_socket())
	{
		return 1;
	}
	SOCKET serfd = creatservesocket();
	if (serfd == INVALID_SOCKET)
	{
		close_socket();
		return 1;
	}
	cout << "Wait for connecting..." << endl;
	SOCKET clifd = accept(serfd, NULL, NULL);
	if (clifd == INVALID_SOCKET)
	{
		err("accept");
		closesocket(serfd);
		close_socket();
		return 1;
	}
	cout << "开始接受矩阵数据" << endl;
	int hang = 0, lie = 0;
	int recv_bytes = 0;
	while (recv_bytes < sizeof(hang))
	{
		int ret = recv(clifd, reinterpret_cast<char*>(&hang) + recv_bytes, sizeof(hang) - recv_bytes, 0);
		if (ret <= 0)
		{
			err("recv data");
		}
		if (ret == SOCKET_ERROR)
		{
			
			int err_code = WSAGetLastError();
			if (err_code == WSAECONNRESET)
			{
				cerr << "连接被对方重置！" << endl;
				cleanup(serfd, clifd);
				return 1;
			}
			
		}
		recv_bytes += ret;
	}
	hang = ntohl(hang);
	recv_bytes = 0;
	while (recv_bytes < sizeof(lie))
	{
		int ret = recv(clifd, reinterpret_cast<char*>(&lie) + recv_bytes, sizeof(lie) - recv_bytes, 0);
		if (ret <= 0)
		{
			err("recv data");
		}
		if (ret == SOCKET_ERROR)
		{
			
			int err_code = WSAGetLastError();
			if (err_code == WSAECONNRESET)
			{
				cerr << "连接被对方重置！" << endl;
				cleanup(serfd, clifd);
				return 1;
			}
			
		}
		recv_bytes += ret;
	}
	lie = ntohl(lie);
	cout << "接收矩阵维度: " << hang << "*" << lie << endl;
	vector<float> data(hang * lie);
	fill(data.begin(), data.end(), 0.0f);
	int total_received = 0;
	while (total_received < data.size()*sizeof(float))
	{
		int ret = recv(clifd, reinterpret_cast<char*>(data.data()) + total_received, data.size() * sizeof(float) - total_received, 0);
		if (ret <= 0)
		{
			err("recv data");
			break;
		}
		if (ret == SOCKET_ERROR)
		{
			
			int err_code = WSAGetLastError();
			if (err_code == WSAECONNRESET)
			{
				cerr << "连接被对方重置！" << endl;
				cleanup(serfd, clifd);
				return 1;
			}
			
		}
		total_received += ret;
	}
	if (total_received != data.size() * sizeof(float)) 
	{
		cerr << "错误：预期接收 " << data.size() * sizeof(float)
			<< " 字节，实际接收 " << total_received << " 字节" << endl;
	
	}
	for (int i = 0; i < data.size(); i++) 
	{
		uint32_t net_val;
		memcpy(&net_val, &data[i], sizeof(net_val));
		float local_val;
		*reinterpret_cast<uint32_t*>(&local_val) = ntohl(net_val); 
		data[i] = local_val;
	}
		cout << "接收矩阵数据成功" << endl;
		Juzhen<float> input;
		buildjuzhen(data.data(), hang, lie,input);
		model_father<float>* model_ptr = create_model<float>("C:\\Users\\Aaron\\Desktop\\GKD-Software-2025-Test-main\\mnist-fc\\");
		if (model_ptr == nullptr)
		{
			cerr << "模型加载失败" << endl;
			closesocket(serfd);
			closesocket(clifd);
			close_socket();
			return 1;
		}
		Juzhen<float> final = model_ptr->forward(input);
		float* senddata = sendjuzhen(final);
		int arr[2] = { htonl(final.hang), htonl(final.lie) };
		int arr_size = sizeof(arr);
		int sent = 0;
		while (sent < arr_size) 
		{
			int ret = send(clifd, (char*)arr + sent, arr_size - sent, 0);
			if (ret <= 0) 
			{
				err("send arr");
				break;
			}
			sent += ret;
		}
		int data_size = final.hang * final.lie * sizeof(float);
		sent = 0;
		while (sent < data_size) {
			int ret = send(clifd, (char*)senddata + sent, data_size - sent, 0);
			if (ret <= 0) {
				err("send data");
				break;
			}
			sent += ret;
		}
		delete[] senddata;
		cout << "已发送矩阵数据" << endl;
		closesocket(serfd);
		closesocket(clifd);
		close_socket();
		return 0;
	}


