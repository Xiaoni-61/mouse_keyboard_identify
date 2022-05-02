#include <iostream>
#include <cstdio>
#include <cstring>
#include <unordered_map>
#include <fstream>
#include<string>
#include<io.h>
#include <direct.h>
#include <vector>
#include <windows.h>

#pragma warning(disable:4996)
using namespace std;

typedef long long LL;
const int N = 120, M = 20000, COUNT = 100; //COUNT:文件数量
int START;

unordered_map<string, int> keytonum; //不同按键对应数组下标
int truesize = 0; //有效数据数量
LL Time[M]; //有效时间
char key[M][20]; //有效按键
char merge_path[100]; //merge文件路径
string path;

//将所有数据文件合并为一个文件，并将','用' '代替，方便读取
void mergedata() {
    /*
    cout << "please input the file path:\n";
    cin >> path;
    cout << "please input the start time:\n";
    cin >> START;
    */
    strcpy(merge_path, path.c_str());
	strcat(merge_path, "\\merge.txt");
    FILE* write = fopen(merge_path, "w");

    //strcpy(merge_path, "merge.txt");
    //FILE* write = fopen(merge_path, "w");
    for (int i = 1; i <= COUNT; i++) {
        ifstream datafile(path + "\\" + to_string(i) + ".txt");
        string s;
        while (getline(datafile, s)) {
            int j;
            for (j = 0; j < s.size(); j++) {
                if (s[j] == ',') s[j] = ' ';
                if (s[j] == 's' && s[j + 1] == 'y' && s[j + 2] == 's' && s[j + 3] == ' ') {
                    int k;
                    for (k = j; k + 4 < s.size(); k++)
                        s[k] = s[k + 4];
                    s[k] = '\0';
                    s[k + 1] = '\0';
                    s[k + 2] = '\0';
                }
            }
            fprintf(write, "%s\n", s.c_str());
        }
    }
    fprintf(write, "\n");
    fclose(write);
}

//建立按键和数组下标的映射
void mapping() {
    ifstream hashfile("hash.txt");
    string key11;
    int cnt = 0;
    while (getline(hashfile, key11)) {
        keytonum[key11] = cnt++;
    }
	
}

//读取合并后的文件
void readdata() {
    int cnt = 0;
    LL Timetmp[M]; //记录所有时间
    char tmp[5], op[M][5], keytmp[M][20]; //tmp用来去除"key"的干扰，op记录"dn"/"up"，keytmp记录所有按键

    memset(Time, 0, sizeof Time);
	//strcat(merge_path, "\\merge.txt");//自己加的！！
    FILE* read = fopen(merge_path, "r");
    //FILE* read = fopen("merge.txt", "r");
    while (!feof(read) && cnt < M) {
		//因为已经处理掉逗号，现在进行每行四个元素的分离
        if (fscanf(read, "%lld %s %s %s", &Timetmp[cnt], tmp, op[cnt], keytmp[cnt]) != EOF) {
            if (!strcmp(keytmp[cnt], "←"))
                strcpy(keytmp[cnt], "LeftArrow");
            if (!strcmp(keytmp[cnt], "→"))
                strcpy(keytmp[cnt], "RightArrow");
            if (!strcmp(keytmp[cnt], "↓"))
                strcpy(keytmp[cnt], "DownArrow");
            if (!strcmp(keytmp[cnt], "↑"))
                strcpy(keytmp[cnt], "UpArrow");
            cnt++;
        }
    }
    fclose(read);

    //将'<'还原为",<"，将"空格键"还原为"SPACE"
    for (int i = 0; i < cnt; i++) {
        if (keytmp[i][0] == '<')
            strcpy(keytmp[i], ",<");
        else if (!strcmp(keytmp[i], "空格键 "))
            strcpy(keytmp[i], "SPACE");
        //cout << keytmp[i] << endl;
    }

    //记录有效数据
    for (int i = 0; i + 1 < cnt; i++) {
        //第一个按键为"dn"，且第二个按键为"up"，且两组数据的key相同，则记录该组数据
        if (!strcmp(op[i], "dn") && !strcmp(op[i + 1], "up") && !strcmp(keytmp[i], keytmp[i + 1])) {
            Time[truesize] = Timetmp[i];
            Time[truesize + 1] = Timetmp[i + 1];
            strcpy(key[truesize], keytmp[i]);
            strcpy(key[truesize + 1], keytmp[i + 1]);
            i++;
            truesize += 2;
        }
    }
}

//输出数据
void outputdata(LL st, LL ed, int user, int delta, string filename, char username[10] ,char eventname[10]) { //四个参数分别为：开始时间、结束时间、用户信息、时间片
    LL DT[N], FAT[N][N], FTC[N][N], FTB[N][N], FTD[N][N], cnt[N][N]; //cnt记录a->b的出现的次数，用于计算平均值

    memset(DT, 0, sizeof DT);		//某个键 从按下到抬起的时间
    memset(FAT, 0, sizeof FAT);
    memset(FTC, 0, sizeof FTC);
    memset(FTB, 0, sizeof FTB);
    memset(FTD, 0, sizeof FTD);
    memset(cnt, 0, sizeof cnt);

    //计算DT,分时间段（在现有给出的时间段中st到ed）
      for (int i = 0; i < truesize; i += 2) {
        if (Time[i] < st) continue; //时间还未到开始时间则跳过
        if (Time[i] > ed) break; //时间已经超过结束时间则终止

        int a = keytonum[key[i]];
        DT[a] = (DT[a] * cnt[a][a] + Time[i + 1] - Time[i]) / (cnt[a][a] + 1); //计算DT平均值
        cnt[a][a] ++; //a->a出现次数+1
    }

    memset(cnt, 0, sizeof cnt); //重置cnt为0，消除计算DT时的干扰

    for (int i = 0; i + 3 < truesize; i += 2) {
        if (Time[i] < st) continue; //时间还未到开始时间则跳过
        if (Time[i] > ed) break; //时间已经超过结束时间则终止
        if (Time[i + 2] - Time[i + 1] >= 2000) continue; //去噪: FAT >= 2000时为无效数据，跳过

        int a = keytonum[key[i]], b = keytonum[key[i + 2]]; //找到相邻两按键对应的数组下标
        FAT[a][b] = (FAT[a][b] * cnt[a][b] + Time[i + 2] - Time[i + 1]) / (cnt[a][b] + 1);
        FTC[a][b] = (FTC[a][b] * cnt[a][b] + Time[i + 2] - Time[i]) / (cnt[a][b] + 1);
        FTB[a][b] = (FTB[a][b] * cnt[a][b] + Time[i + 3] - Time[i + 1]) / (cnt[a][b] + 1);
        FTD[a][b] = (FTD[a][b] * cnt[a][b] + Time[i + 3] - Time[i]) / (cnt[a][b] + 1);
        cnt[a][b] ++; //a->b出现次数+1
    }

    //输出数据到文件中
    FILE* write = fopen(filename.c_str(), "a+");
    // fprintf(write, "%d %d\n", user, delta); //前两维输出用户信息和时间片
    //输出所有按键的DT值
    for (int i = 0; i < 110; i++) {
        fprintf(write, "%lld ", DT[i]);
    }
    //fprintf(write, "\n");
    //输出每个按键
    for (int i = 0; i < 110; i++) {
        //首先输出当前按键对应所有按键的FAT值
        for (int j = 0; j < 110; j++) {
            fprintf(write, "%lld ", FAT[i][j]);
        }
        //fprintf(write, "\n");
        //其次输出当前按键对应所有按键的FTC值，并以此类推
        for (int j = 0; j < 110; j++) {
            fprintf(write, "%lld ", FTC[i][j]);
        }
        //fprintf(write, "\n");
        for (int j = 0; j < 110; j++) {
            fprintf(write, "%lld ", FTB[i][j]);
        }
        //fprintf(write, "\n");
        for (int j = 0; j < 110; j++) {
            fprintf(write, "%lld ", FTD[i][j]);
        }
        //fprintf(write, "\n");
    }
    fprintf(write, "%s %s\n", username, eventname); //输出完该组数据，换行
	// fprintf(write, "\n", username); //输出完该组数据，换行

    fclose(write);

}


string UTF8ToGB(const char* str)
{
    string result;
    WCHAR* strSrc;
    LPSTR szRes;

    //获得临时变量的大小
    int i = MultiByteToWideChar(CP_UTF8, 0, str, -1, NULL, 0);
    strSrc = new WCHAR[i + 1];
    MultiByteToWideChar(CP_UTF8, 0, str, -1, strSrc, i);

    //获得临时变量的大小
    i = WideCharToMultiByte(CP_ACP, 0, strSrc, -1, NULL, 0, NULL, NULL);
    szRes = new CHAR[i + 1];
    WideCharToMultiByte(CP_ACP, 0, strSrc, -1, szRes, i, NULL, NULL);

    result = szRes;
    delete[]strSrc;
    delete[]szRes;

    return result;
}



int main()
{
    string check = "50people";
    // check = UTF8ToGB(check.c_str());
    string file_path;
    cout << "please input file address:\n";
    cin >> file_path;
    ifstream in(file_path);
    string line;
    string splitpath;
    if (in) // 有该文件
    {
		mapping();
        while (getline(in, line)) // line中不包括每行的换行符
        {
            truesize = 0;
            //memset(Time, 0, sizeof(Time));
            //memset(key, 0, sizeof(key));
            path = line;
            splitpath = path;
            const char* sep = "\\"; //可按多个字符来分割
            char p[10][10];
            char* save = NULL;
            memset(p, 0, sizeof(char) * 100);//对p这个二维数组进行初始化 0
            strcpy(p[0], strtok((char*)splitpath.c_str(), sep)); //用sep进行分解，之后把值赋给p[0]
            for (int i = 1; i < 10; i++) {
                save = strtok(NULL, sep);
                if (save == NULL)
                    break;
                strcpy(p[i], save);
            }
            if (strcmp(p[5], check.c_str()) != 0) {
                // splitpath = path = UTF8ToGB(line.c_str()).c_str();
                memset(p, 0, sizeof(char) * 100);
                strcpy(p[0], strtok((char*)splitpath.c_str(), sep));
                for (int i = 1; i < 10; i++) {
                    save = strtok(NULL, sep);
                    if (save == NULL)
                        break;
                    strcpy(p[i], save);
                }
            }
            //path = "D:\\大学生活\\大四下\\计算机毕设\\数据\\50people\\于高远\\微博键盘";
            getline(in, line);
            START = atoi(line.c_str()); //把字符型等等数转化成整数
            //START = 183227281;

            mergedata();

            readdata();

            //st为鼠标开始记录的时间，ed为经过10分钟后结束的时间
            LL st = START, ed = st + 1000 * 10 * 60;

            //用户姓名
            /*
            const char* sep = "\\"; //可按多个字符来分割
            char p[10][10];
            char* save = NULL;
            memset(p, 0, sizeof(char) * 100);
            strcpy(p[0], strtok((char*)path.c_str(), sep));
            for (int i = 1; i < 10; i++) {
                save = strtok(NULL, sep);
                if (save == NULL)
                    break;
                strcpy(p[i], save);
            }
            */
            //delta中记录时间片，可根据自己需求进行更改
            //int all_overlapping[20] = { 0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95 };
            int all_overlapping[1] = { 0 };
            //遍历每一种时间片
			mkdir("data");
            for (int interval = 10 ; interval < 61; interval+=10)  //时间片
                for (int k = 0; k < 1; k++) {
                    string dir = "data/" + string(p[7]) + "_" + string(p[6]) + "_" + to_string(interval) + "s";
					if (_access(dir.c_str(), 0) == -1)
						if (mkdir(dir.c_str()) == 0)
							printf("successfully create!\n");
                    string filename = "./data/" + string(p[7]) + "_" + string(p[6]) + "_" + to_string(interval) + "s/" + string(p[7]) + "_" + string(p[6]) + "_" + to_string(interval) + "s.txt";
                    // string filename = "./data/" + string(p[3]) + "_" + to_string(interval) + "s/" + string(p[2]) + "_" + string(p[3]) + "_" + to_string(interval) + "s_" + to_string(all_overlapping[k] * 100) + "%.txt";
                    //string filename = "./" + string("记事本键盘_") + to_string(delta[k]) + "s/" + string(p[2]) + "_" + string("记事本键盘_") + to_string(delta[k]) + "s.txt";
                    //遍历开始时间到结束时间，按时间片的长度递增
                    for (LL i = st; i < ed; i += (1000 * interval * (1 - all_overlapping[k]))) {
                        //以当前时间为开始时间，以当前时间+时间片大小为结束时间，进行数据输出
                        outputdata(i, i + 1000 * interval, 1, interval, filename, p[6], p[7]);
                        //打印当前的时间片以及开始后秒数
                        cout << p[6] << '_' << p[7] << ' ' <<interval << ' ' << (i - st) / 1000 << endl;
                    }
                }
        }
    }
    else // 没有该文件
    {
        cout << "no such file" << endl;
    }
    return 0;
}
