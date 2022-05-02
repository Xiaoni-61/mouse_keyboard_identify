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
第一列512代表移动，
513代表左键按下，
514代表左键释放，
516代表右键按下，
517代表右键释放，
522代表滚轮，
*/

typedef struct
{
	int moveID;
	unsigned long timestamp;	//此时时间
	int x;
	int y;
}movement;						//原始数据

typedef  struct dnode *dptr;
struct dnode					//原始数据链表
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