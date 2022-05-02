// mousedata_prcs.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include <Windows.h>
#include <cstdlib>
#define  _USE_MATH_DEFINES 1

#include <cmath>
#include "MouseFirst.h"
#include "winsock.h"
//#include <mysql.h> 
#include <string>
#include <cstdio>
#include <iostream>
#include <cstring>
#include <afxwin.h>
#include <atlstr.h>
#include <stdio.h>
#include <time.h>
#define TIMESPAN 1
#define MAX_USER 4

#define M_PI_4 0.785398163397448309616

using namespace std;
char* userID;
//MYSQL mysql;
int id;
int max_index[]={60,60,60,60};
int current_user = 0;
string filepath;
string outpath;

int get_mouse_speed()
{
	int sp;
	SystemParametersInfo(SPI_GETMOUSESPEED,
		0,
		&sp,
		0);
	printf("mouse speed:%d\n", sp);
	return sp;
}
void insert(_res tmp, int index)
{
	//	printf("%d,%lu,%lu\t%d,%d,index:%d\n", tmp.eventID, tmp.starttime, tmp.endtime, tmp.direction, (int)tmp.distance,index);
		//char* sql = (char*)malloc(300);
	float speed;
	FILE* fpp;
	//memset(sql, 0, 300);
	if (tmp.endtime > tmp.starttime)
		speed = (float)1000.0 * tmp.distance / (tmp.endtime - tmp.starttime);
	else
		speed = 0;
	string fpath;
	fpath = outpath + "\\out.txt";
	//cout << fpath << endl;
	//printf("%d,%d,%d,%d,%f,%d,%lu,%lu,'%s',%d\n", id, tmp.eventID, (int)tmp.distance, tmp.endtime - tmp.starttime, speed, tmp.direction, tmp.starttime, tmp.endtime, userID, index);
	fopen_s(&fpp, fpath.c_str(), "a+");
	fprintf_s(fpp, "%d,%d,%d,%d,%f,%d,%lu,%lu,'%s',%d\n", id, tmp.eventID, (int)tmp.distance, tmp.endtime - tmp.starttime, speed, tmp.direction, tmp.starttime, tmp.endtime, userID, index);
	fclose(fpp);
	//sprintf(sql, "insert into 5_people values(%d,%d,%d,%d,%f,%d,%lu,%lu,'%s',%d)", id, tmp.eventID, (int)tmp.distance, tmp.endtime - tmp.starttime, speed, tmp.direction, tmp.starttime, tmp.endtime, userID, index);
	//printf("%s\n", sql);
	/*if (mysql_query(&mysql, sql))
	{
		printf("add failed:%d\t%s:\n", mysql_errno(&mysql), mysql_error(&mysql));

	}
	else
	{
		//printf("add succeed\n");
		id++;
	}*/
}
/*
判断是单击还是双击还是拖拽
单击，双击的部分会在这里入库
是拖拽会在原函数继续循环
return 0不是单击不是双击
1单击
2双击
*/
int left_click(dptr h, int index) 
{
	int dclick_itvl = ::GetDoubleClickTime();//根据系统设置不同，其实无卵用，就看我的系统
	dclick_itvl = DCLICK_ITVL;
	if (h->mdata.moveID != WM_LBUTTONDOWN)
	{
		printf("left click:input error\n");
		return 0;
	}
	else
	{
		if (h->next&&h->next->next&&h->next->next->next)
		{
			dptr m1 = h;
			dptr m2 = h->next;
			dptr m3 = h->next->next;
			dptr m4 = h->next->next->next;
			int t1 = m2->mdata.timestamp - m1->mdata.timestamp;
			int t2 = m3->mdata.timestamp - m2->mdata.timestamp;
			int t3 = m4->mdata.timestamp - m3->mdata.timestamp;
			if (m2->mdata.moveID == WM_LBUTTONUP&&m3->mdata.moveID == WM_LBUTTONDOWN&&m4->mdata.moveID == WM_LBUTTONUP&&t1 < dclick_itvl&&t2 < dclick_itvl&&t3 < dclick_itvl)//双击
			{
				_res tmp;
				tmp.direction = 0;
				tmp.distance = 0;
				tmp.eventID = DCLICK;
				tmp.starttime = h->mdata.timestamp;
				tmp.endtime = m4->mdata.timestamp;
				insert(tmp,index);
				return 2;
			}
			else if (m2->mdata.moveID == WM_LBUTTONUP)//单击
			{
				_res tmp;
				tmp.direction = 0;
				tmp.distance = 0;
				tmp.eventID = LCLICK;
				tmp.starttime = h->mdata.timestamp;
				tmp.endtime = m2->mdata.timestamp;
				insert(tmp,index);
				return 1;
			}
			else //拖拽
			{
				return 0;
			}
		}
		else if(h->next->mdata.moveID == WM_LBUTTONUP)//无第二次按键，单击
		{
			_res tmp;
			tmp.direction = 0;
			tmp.distance = 0;
			tmp.eventID = LCLICK;
			tmp.starttime = h->mdata.timestamp;
			tmp.endtime = h->next->mdata.timestamp;
			insert(tmp,index);
			return 1;
		}
		else//下一个是mm，拖拽
		{
			return 0;
		}
		

	}
	return 0;
}

void print_list(dptr h)
{	
	while (h->next)
	{
		printf("mid:%d,time:%lu,x:%d,y:%d\n", h->next->mdata.moveID, h->next->mdata.timestamp, h->next->mdata.x, h->next->mdata.y);
		h = h->next;
	}
}

int direct(int x0,int y0,int xt,int yt)
{
	if(x0==xt&&y0==yt)
		return 0;
	else 
	{
		xt = xt - x0;
		yt = y0 - yt;
		int res = 0;
		double angle = atan2(yt,xt);  //弧度
		res = angle / M_PI_4;
		if (res <= -4) res = -3;
		else if (res > 3) res = 3;
		//printf("%d，%d,%f,%d\n", xt, yt, angle, res);
		switch (res) 
		{
			case 1:return 1;
			case 2:return 8;
			case 3:return 7;
			case -3:return 6;
			case -2:return 5;
			case -1:return 4;
			case 0: {if (yt < 0) return 3;else return 2;}
			default:return -1;	//理论上没有-1
		}
	}
}
double dist(int x0, int y0, int xt, int yt)
{
	double a = (double)((x0 - xt)*(x0 - xt) + (y0 - yt)*(y0 - yt));
	return sqrt(a);
}
/*
给一个当前为mm的指针
识别一段连续的移动
返回一个gm_res结构体
结束时gm_res结构体中指针指向gm的最后一个mm节点
*/
gm_res general_move(dptr h)
{
	gm_res tmp;
	tmp.starttime = h->mdata.timestamp;
	tmp.distance = 0;
	int x0 = h->mdata.x;
	int y0 = h->mdata.y;
	double distance = 0;
	while (h->next!=NULL && h->next->mdata.moveID==WM_MOUSEMOVE)
	{
		if (h->next->mdata.timestamp - h->mdata.timestamp > TIME_INTVL)
		{
			//超时，结束判定
			tmp.distance = distance;
			tmp.endtime = h->mdata.timestamp;
			tmp.direction = direct(x0, y0, h->mdata.x, h->mdata.y);
			tmp.last_mm = h;
			return tmp;
		}
		else
		{
			//h->next依然算在此次事件中，计算h和h->next节点的距离等
			distance += dist(h->mdata.x, h->mdata.y, h->next->mdata.x, h->next->mdata.y);
			h = h->next;
		}
	}
	tmp.distance = distance;
	tmp.direction = direct(x0, y0, h->mdata.x, h->mdata.y);	
	tmp.endtime = h->mdata.timestamp;
	tmp.last_mm = h;
	//返回的tmp.distance=0,指针还是原来的不变，direction=0，则说明只有一个mm，后面处理就不算了。
	return tmp;

}
/*
计算拖拽之间的移动的距离
返回结构体中，只有距离有用
还要再加上开头和结尾点击时的移动距离
*/
gm_res drag_move(dptr h)
{
	gm_res final_res;
	final_res.distance = 0;
	gm_res tmp;
	int dm_stop = 0;
	while (!dm_stop)
	{
		tmp = general_move(h);
		final_res.distance += tmp.distance;
		final_res.last_mm = tmp.last_mm;
		if (tmp.last_mm->next->mdata.moveID == WM_MOUSEMOVE) h=tmp.last_mm->next;
		else dm_stop = 1;		
	}
	return final_res;
	
}
/*
过滤反人类操作
flag用于过滤鼠标左（右）键按下时，右（左）键跟着按下的情况，将第一个按下到最后一个抬起之间的数据过滤；
*/
/*
设想一种场景:
MOVE->lButtonDown->rLuttonDown->lButtonUp->move->rButtonUp
此时会清除从第一个move开始的全部信息，到全部键抬起来为止
*/
void filter(dptr h)
{
	int lbutton=0;
	int rbutton=0;
	int flag=0;
	dptr hl = h;
	while (h->next)
	{
		h = h->next;
		if (lbutton == 0 && rbutton == 0)
		{
			if(h->mdata.moveID==WM_MOUSEMOVE)
				hl = h;
			else if (h->mdata.moveID == WM_LBUTTONDOWN)
			{
				//hl不变
				lbutton = 1;
			}
			else if (h->mdata.moveID == WM_LBUTTONUP)
			{
				//impossible
				printf("filter:input error\n");
			}
			else if (h->mdata.moveID == WM_RBUTTONDOWN)
			{
				//hl不变
				rbutton = 1;
			}
			else if (h->mdata.moveID == WM_RBUTTONUP)
			{
				//impossible
				printf("filter:input error\n");
			}
		}
		else if (lbutton == 1 && rbutton == 0)
		{
			if (h->mdata.moveID == WM_MOUSEMOVE)
			{
				//null
			}
			else if (h->mdata.moveID == WM_LBUTTONDOWN)
			{
				//impossible
				printf("filter:input error\n");
			}
			else if (h->mdata.moveID == WM_LBUTTONUP)
			{
				if (flag == 1)
				{
					if (h->next)
					{
						flag = 0;
						lbutton = 0;
						hl->next = h->next;
					}
					else hl->next = NULL;
				}
				else
				{
					hl = h;
					lbutton = 0;
				}
			}
			else if (h->mdata.moveID == WM_RBUTTONDOWN)
			{
				//hl不变
				rbutton = 1;
				flag = 1;
			}
			else if (h->mdata.moveID == WM_RBUTTONUP)
			{
				//impossible
				printf("filter:input error\n");
			}
		}
		else if (lbutton == 0 && rbutton == 1)
		{
			if (h->mdata.moveID == WM_MOUSEMOVE)
			{
				//null
			}
			else if (h->mdata.moveID == WM_LBUTTONDOWN)
			{
				lbutton = 1;
				flag = 1;
			}
			else if (h->mdata.moveID == WM_LBUTTONUP)
			{
				//impossible
				printf("filter:input error\n");
			}
			else if (h->mdata.moveID == WM_RBUTTONDOWN)
			{
				//impossible
				printf("filter:input error\n");
			}
			else if (h->mdata.moveID == WM_RBUTTONUP)
			{
				if (flag == 1)
				{
					if (h->next)
					{
						flag = 0;
						rbutton = 0;
						hl->next = h->next;
					}
					else hl->next = NULL;
				}
				else
				{
					hl = h;
					rbutton = 0;
				}
				
			}
		}
		else if (lbutton == 1 && rbutton == 1)
		{
			if (h->mdata.moveID == WM_MOUSEMOVE)
			{
				//null
			}
			else if (h->mdata.moveID == WM_LBUTTONDOWN)
			{
				//impossible
				printf("filter:input error\n");
			}
			else if (h->mdata.moveID == WM_LBUTTONUP)
			{
				lbutton = 0;
			}
			else if (h->mdata.moveID == WM_RBUTTONDOWN)
			{
				//impossible
				printf("filter:input error\n");
			}
			else if (h->mdata.moveID == WM_RBUTTONUP)
			{
				rbutton = 0;
			}
		}
	}
	hl->next = NULL;
}

/*
从头结点开始，遍历一遍分出
GENERAL_MOVE 1
DRAG_DROP 2
MOVE_LCLICK 3
MOVE_RCLICK 4
MOVE_DCLICK 5

LCLICK 7
RCLICK 8
DCLICK 9
PAUSE_CLICK 10
NONE_EVENT 0
*/
void classify(dptr h, int index)	//传入为原始数据链表节点
{
	_res tmp;						//最后写入的格式
	dptr h_before;
	dptr head = h;
	tmp.eventID = NONE_EVENT;
	
	
	while (h->next)
	{
		h_before = h;
		h = h->next;
		if (tmp.eventID == NONE_EVENT)
		{
			if (h->mdata.moveID == WM_MOUSEMOVE)
			{
				gm_res gm;
				if ((!h->next || h->next->mdata.moveID != WM_MOUSEMOVE || h) && h_before!=head && h->mdata.timestamp-h_before->mdata.timestamp <= TIME_INTVL)
					gm = general_move(h_before);
				else
					gm = general_move(h);
				
				tmp.direction = gm.direction;
				tmp.distance = gm.distance;
				tmp.starttime = gm.starttime;
				tmp.endtime = gm.endtime;

				h = gm.last_mm;
				if (!h->next)
				{
					continue;
				}
				else//有h->next
				{
					if (h->next->mdata.moveID == WM_MOUSEMOVE)
					{
						//tmp.userID;
						tmp.eventID = GENERAL_MOVE;
						insert(tmp,index);

						tmp.eventID = NONE_EVENT;
					}
					else if (h->next->mdata.moveID == WM_LBUTTONDOWN)
					{
						//移动点击时间差距太大，算作两次事件,此次是gm
						if (h->next->mdata.timestamp - h->mdata.timestamp > PAUSE_CLICK_INTVL)
						{
							tmp.eventID = GENERAL_MOVE;
							insert(tmp,index);

							tmp.eventID = NONE_EVENT;
						}
						//pause_click
						else
						{
							int s_or_d = left_click(h->next,index);
							if (s_or_d == 0)
							{
								//tmp.userID;
								tmp.eventID = GENERAL_MOVE;
								insert(tmp,index);

								tmp.eventID = NONE_EVENT;
							}
							else if (s_or_d == 1) //单击
							{
								//先入move_lclick
								tmp.eventID = MOVE_LCLICK;
								tmp.endtime = h->next->next->mdata.timestamp;
								insert(tmp,index);

								//pauseclick
								tmp.direction = 0;
								tmp.distance = 0;
								tmp.eventID = PAUSE_CLICK;
								tmp.starttime = h->mdata.timestamp;
								tmp.endtime = h->next->mdata.timestamp;
								insert(tmp,index);

								h = h->next->next;
								tmp.eventID = NONE_EVENT;
							}
							else //双击
							{
								tmp.eventID = MOVE_DCLICK;
								tmp.endtime = h->next->next->next->next->mdata.timestamp;
								insert(tmp,index);

								tmp.direction = 0;
								tmp.distance = 0;
								tmp.eventID = PAUSE_CLICK;
								tmp.starttime = h->mdata.timestamp;
								tmp.endtime = h->next->mdata.timestamp;
								insert(tmp,index);
								
								h = h->next->next->next->next;
								tmp.eventID = NONE_EVENT;
							}
						}
					}
					else if (h->next->mdata.moveID == WM_RBUTTONDOWN)
					{
						if (h->next->mdata.timestamp - h->mdata.timestamp > PAUSE_CLICK_INTVL)//超时，算作gm
						{
							tmp.eventID = GENERAL_MOVE;
							insert(tmp,index);

							tmp.eventID = NONE_EVENT;
						}
						else if (h->next->next->mdata.moveID == WM_RBUTTONUP)//右单击
						{
							//先插入move_rclick
							tmp.endtime = h->next->next->mdata.timestamp;
							tmp.eventID = MOVE_RCLICK;
							insert(tmp,index);

							//插入rclick
							tmp.starttime= h->next->mdata.timestamp;
							tmp.endtime = h->next->next->mdata.timestamp;
							tmp.eventID = RCLICK;
							tmp.direction = 0;
							tmp.distance = 0;
							insert(tmp,index);

							//插入pause_click
							tmp.starttime = h->mdata.timestamp;
							tmp.endtime = h->next->mdata.timestamp;
							tmp.eventID = PAUSE_CLICK;
							tmp.direction = 0;
							tmp.distance = 0;
							insert(tmp,index);

							tmp.eventID = NONE_EVENT;
							h = h->next->next;
						}
						else //移动，下面是右键拖拽了，所以先入了gm
						{
							tmp.eventID = GENERAL_MOVE;
							insert(tmp,index);

							tmp.eventID = NONE_EVENT;
						}
					}
					else printf("error:mm next is not mm or lbd or rbd.\n");
				}
			}
			else if (h->mdata.moveID == WM_LBUTTONDOWN)
			{
				int s_or_d = left_click(h,index);
				if (s_or_d == 0)//drag and drop
				{
					gm_res gm = drag_move(h->next);
					if (gm.last_mm->next->mdata.moveID == WM_LBUTTONUP)
					{
						double distance = dist(h->mdata.x, h->mdata.y, h->next->mdata.x, h->next->mdata.y) + gm.distance + dist(gm.last_mm->mdata.x, gm.last_mm->mdata.y, gm.last_mm->next->mdata.x, gm.last_mm->next->mdata.y);
						tmp.starttime = h->mdata.timestamp;
						tmp.endtime = gm.last_mm->next->mdata.timestamp;
						tmp.eventID = DRAG_DROP;
						tmp.direction = direct(h->mdata.x, h->mdata.y, gm.last_mm->next->mdata.x, gm.last_mm->next->mdata.y);
						tmp.distance = distance;
						insert(tmp,index);

						tmp.eventID = NONE_EVENT;
						h = gm.last_mm->next;
					}
					else printf("error: left drag end without lbu\n");
				}
				else if (s_or_d == 1)
				{
					tmp.starttime = h->mdata.timestamp;
					tmp.endtime = h->next->mdata.timestamp;
					tmp.eventID = LCLICK;
					tmp.direction = 0;
					tmp.distance = 0;
					//insert(tmp);

					tmp.eventID = NONE_EVENT;
					h = h->next;
				}
				else if (s_or_d == 2)
				{
					tmp.starttime = h->mdata.timestamp;
					tmp.endtime = h->next->next->next->mdata.timestamp;
					tmp.eventID = DCLICK;
					tmp.direction = 0;
					tmp.distance = 0;
					//insert(tmp);

					tmp.eventID = NONE_EVENT;
					h = h->next->next->next;
				}
				else printf("error: left_click return not 0,1,2\n");
			}
			else if (h->mdata.moveID == WM_RBUTTONDOWN)
			{
				if (!h->next) printf("error:end with rbd\n");
				else
				{
					if (h->next->mdata.moveID == WM_RBUTTONUP)
					{
						tmp.direction = 0;
						tmp.distance = 0;
						tmp.starttime = h->mdata.timestamp;
						tmp.endtime = h->next->mdata.timestamp;
						tmp.eventID = RCLICK;
						insert(tmp,index);

						tmp.eventID = NONE_EVENT;
						h = h->next;
					}
					else if (h->next->mdata.moveID == WM_MOUSEMOVE)
					{
						gm_res gm = drag_move(h->next);
						double distance = dist(h->mdata.x, h->mdata.y, h->next->mdata.x, h->next->mdata.y) + gm.distance + dist(gm.last_mm->mdata.x, gm.last_mm->mdata.y, gm.last_mm->next->mdata.x, gm.last_mm->next->mdata.y);
						tmp.starttime = h->mdata.timestamp;
						tmp.endtime = gm.last_mm->next->mdata.timestamp;
						tmp.eventID = DRAG_DROP;
						tmp.direction = direct(h->mdata.x, h->mdata.y, gm.last_mm->next->mdata.x, gm.last_mm->next->mdata.y);
						tmp.distance = distance;
						insert(tmp,index);

						tmp.eventID = NONE_EVENT;
						h = gm.last_mm->next;
					}
					else printf("error: rbd is not followed by mm or  rbu");
				}
			}
			else printf("error: after none_event, not mm or lbd or rbd\n");
		}
		else
		{
			printf("classify: eventID != NONE_EVENT error\n");
			//不可能了，都在上面处理完。
		}
	}
}
int silence(dptr h, int index)
{
	
	unsigned long t1=h->next->mdata.timestamp;
	while (h->next)
	{
		if (h->next->mdata.timestamp - t1 > TIME_INTVL)
		{
			_res tmp;
			tmp.direction = 0;
			tmp.distance = 0;
			tmp.endtime = h->next->mdata.timestamp;
			tmp.starttime = t1;
			tmp.eventID = SILENCE;
			insert(tmp,index);
		}
		t1 = h->next->mdata.timestamp;
		h = h->next;
	}
	return 0;
}

void FindAllFile(CString strFoldername)  
{   
    CFileFind tempFind;   
    BOOL bFound; //判断是否成功找到文件 
	CTime ftime = NULL;
	CTime currentTime = NULL;
	int index = 1;

    bFound=tempFind.FindFile(strFoldername + _T("\\*.*"));   //修改" "内内容给限定查找文件类型  
	if(bFound==false)
		cout<<"can't find"<<endl;
    CString strTmp;   //如果找到的是文件夹 存放文件夹路径  
    while(bFound && index<=max_index[current_user])      //遍历所有文件  
    {   
        bFound=tempFind.FindNextFile(); //第一次执行FindNextFile是选择到第一个文件，以后执行为选择到下一个文件  
        if(tempFind.IsDots())   
            continue; //如果找到的是返回上层的目录 则结束本次查找 
		if (tempFind.GetFileName() == "记事本键盘" || tempFind.GetFileName() == "微博键盘" || tempFind.GetFileName() == "淘宝键盘" || tempFind.GetFileName() == "游戏键盘")
			continue;
        if(tempFind.IsDirectory())   //找到的是文件夹，则遍历该文件夹下的文件  
        {   
			/*
			if (tempFind.GetFileName() != "记事本鼠标" && tempFind.GetFileName() != "微博鼠标" && tempFind.GetFileName() != "淘宝鼠标" && tempFind.GetFileName() != "游戏鼠标") {
				strTmp = tempFind.GetFilePath();
				FindAllFile(strTmp);
				continue;
			}
			*/
			outpath = (LPCSTR)(CStringA)(tempFind.GetFilePath());
			std::cout<<"\n\nDirectory: "<< (LPCSTR)(CStringA)tempFind.GetFileName()<<endl;
			strTmp=tempFind.GetFilePath(); 
			if(tempFind.GetFileName()!="mousemovements")
				//strcpy(userID, tempFind.GetFileName());
				::wsprintfA(userID, "%ls", (LPCTSTR)tempFind.GetFileName());
			FindAllFile(strTmp); 
//			strcpy(userID,"niudanyang");//test
        }   
        else      
		{ 
			tempFind.GetLastWriteTime(currentTime);
			CTimeSpan timespan(0,0,TIMESPAN*index,0);
			if(ftime == NULL)
			{
				ftime = currentTime;
//				tempFind.GetLastWriteTime(ftime);
			}
			
			while(ftime + timespan <= currentTime && index<max_index[current_user])
			{
				
				timespan = CTimeSpan::CTimeSpan(0,0,TIMESPAN*++index,0);
				if(ftime + timespan <= currentTime)
				{
					std::cout<<"\n\nNo file in index: "<<index<<endl;
					//空index处理
					_res blank;
					blank.direction=0;
					blank.distance=0;
					blank.endtime=0;
					blank.eventID=0;
					blank.starttime=0;
					insert(blank,index);
					if(index==max_index[current_user])
						index++;
				}
			}
			if (tempFind.GetFileName() == "out.txt") break;
			std::cout<<"\nGetFileName: "<< (LPCSTR)(CStringA)tempFind.GetFileName()<<" "<<index;
			std::cout<<"\nGetLastWriteTime:"<< (LPCSTR)(CStringA)currentTime.Format("[%Y-%m-%d %H:%M:%S]");

			if(index==max_index[current_user]+1)
				continue;

			//非空index处理
			FILE* fp;
			char temp[100];
			::wsprintfA(temp, "%ls", (LPCTSTR)tempFind.GetFilePath());
			//fp = fopen(tempFind.GetFilePath(), "r");
			fp = fopen(temp, "r");
			int moveID;
			unsigned long timestamp;
			int x, y;
			dptr head = (dptr)malloc(sizeof(dnode)); //空头链表（原始数据）
			dptr lnode;
			lnode = head;
			int node_num = 0;
			int lbutton=0, rbutton=0;
			while (fscanf(fp, "%d", &moveID) != EOF)
				{
				
				if (moveID != WM_MOUSEMOVE && moveID != WM_LBUTTONDOWN && moveID != WM_LBUTTONUP && moveID != WM_RBUTTONDOWN && moveID != WM_RBUTTONUP) 
				{
					printf("%d\n", moveID);
					fscanf(fp, "%lu %d %d", &timestamp, &x, &y);
					continue;//不进行记录
				}
				//过滤没有按下就抬起的按键，按下之后又按下的按键
				else	
				{
					if (lbutton == 0 && moveID == WM_LBUTTONUP)
					{

						fscanf(fp, "%lu %d %d", &timestamp, &x, &y);
						continue;
					}
					else if (rbutton == 0 && moveID == WM_RBUTTONUP)
					{
						fscanf(fp, "%lu %d %d", &timestamp, &x, &y);
						continue;
					}
					else if (lbutton == 1 && moveID == WM_LBUTTONDOWN)
					{
						lbutton = 1;
						fscanf(fp, "%lu %d %d", &timestamp, &x, &y);
						continue;
					}
					else if (rbutton == 1 && moveID == WM_RBUTTONDOWN)
					{
						rbutton = 1;
						fscanf(fp, "%lu %d %d", &timestamp, &x, &y);
						continue;
					}
					else if (lbutton == 0 && moveID == WM_LBUTTONDOWN)
					{
						lbutton = 1;
					}
					else if (rbutton == 0 && moveID == WM_RBUTTONDOWN)
					{
						rbutton = 1;
					}
					else if (lbutton == 1 && moveID == WM_LBUTTONUP)
					{
						lbutton = 0;
					}
					else if (rbutton == 1 && moveID == WM_RBUTTONUP)
					{
						rbutton = 0;
					}
					//其他情况：移动
					fscanf(fp, "%lu %d %d", &timestamp, &x, &y);
					dptr tmp;
					tmp = (dptr)malloc(sizeof(dnode));
					tmp->mdata.moveID = moveID;
					tmp->mdata.timestamp = timestamp;
					tmp->mdata.x = x;
					tmp->mdata.y = y;
					tmp->next = NULL;
					lnode->next = tmp;
					lnode = tmp;
					node_num++;
				}

			}
			fclose(fp);
			filter(head); //此时的head指的是一个文件一个文件中的信息链表

			classify(head,index);
			silence(head,index);
		}
	}
	index++;
	while(ftime!=NULL && index<=max_index[current_user])
	{
		std::cout<<"\n\nNo file in index: "<<index<<endl;
		//空index处理
		_res blank;
		blank.direction=0;
		blank.distance=0;
		blank.endtime=0;
		blank.eventID=0;
		blank.starttime=0;
		insert(blank,index);
		index++;
	}
    tempFind.Close(); 
	if(ftime!=NULL)
		current_user++;

    return;   
}

int main(int argc,char** argv)
{
	userID = (char*)malloc(30);
	id = 1;

	int res, j;




	printf("-------------------------------------------------------------------------------\n");

	string file_path;
	cout << "\nPlease input the file path:\n";
	cin >> file_path;
	filepath = file_path;
	CString path(file_path.c_str());
	//string command;
	//command = "mkdir -p " + filepath;
	//system(command.c_str());
	FindAllFile(path);

	return 0;
}

