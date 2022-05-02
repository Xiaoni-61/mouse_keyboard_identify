#pragma once
#define NONE_EVENT 0
#define GENERAL_MOVE 1
#define DRAG_DROP 2
#define MOVE_LCLICK 3
#define MOVE_RCLICK 4
#define MOVE_DCLICK 5
#define SILENCE 6
#define LCLICK 7
#define RCLICK 8
#define DCLICK 9
#define PAUSE_CLICK 10

#define TIME_INTVL 100
#define PAUSE_CLICK_INTVL 3000
#define DCLICK_ITVL 500

/*
moveID:
��һ��512�����ƶ���
513����������£�
514��������ͷţ�
516�����Ҽ����£�
517�����Ҽ��ͷţ�
522������֣�
*/

typedef struct
{
	int moveID;
	unsigned long timestamp;	//��ʱʱ��
	int x;
	int y;
}movement;						//ԭʼ����

typedef  struct dnode *dptr;
struct dnode					//ԭʼ��������
{
	movement mdata;
	dptr next;
};
typedef  struct silence_res *silence_res_ptr;
struct silence_res
{
	unsigned long starttime;
	unsigned long stoptime;
	int x;
	int y;
	dptr next;
};

struct _res
{
	int eventID;
	char userID[100];
	unsigned long starttime;
	unsigned long endtime;
	double distance;
	int direction;
};
struct gm_res
{
	unsigned long starttime;
	unsigned long endtime;
	int direction;
	double distance;
	dptr last_mm;
};