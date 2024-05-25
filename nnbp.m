function [sys,x0,str,ts,simStateCompliance] = nnbp(t,x,u,flag,nh,xite,alfa)
switch flag
  case 0
    [sys,x0,str,ts,simStateCompliance]=mdlInitializeSizes(nh);
%初始化函数
  case 3
    sys=mdlOutputs(t,x,u,nh,xite,alfa);
%输出函数
  case {1,2,4,9}
    sys=[];
  otherwise
    DAStudio.error('Simulink:blocks:unhandledFlag', num2str(flag));
end
function [sys,x0,str,ts,simStateCompliance]=mdlInitializeSizes(nh)
%调用初始化函数，参数nh确定隐含层层数
sizes = simsizes;
sizes.NumContStates  = 0;
sizes.NumDiscStates  = 0;
sizes.NumOutputs     = 4+6*nh;
%定义输出变量，包括控制变量u,三个PID参数：Kp,Ki,Kd,隐含层+输出层所有加权系数
sizes.NumInputs      = 8+12*nh;
%定义输入变量，包括前8个参数[e(k);e(k-1);e(k-2);y(k);y(k-1);r(k);u(k-1);u(k-2)]
%隐含层+输出层权值系数（k-2),隐含层+输出层权值系数（k-1）
sizes.DirFeedthrough = 1;
sizes.NumSampleTimes = 1; 
sys = simsizes(sizes);
x0  = [];
str = [];
ts  = [0 0];
simStateCompliance = 'UnknownSimState';

function sys=mdlOutputs(t,x,u,nh,xite,alfa)
wi_2 = reshape(u(9:8+3*nh),nh,3);
%隐含层（k-2)权值系数矩阵，维数nh*3
wo_2 = reshape(u(9+3*nh:8+6*nh),3,nh);
%输出层（k-2）权值系数矩阵，维数3*nh
wi_1 = reshape(u(9+6*nh:8+9*nh),nh,3);
%隐含层（k-1)权值系数矩阵，维数nh*3
wo_1 = reshape(u(9+9*nh:8+12*nh),3,nh);
%输出层（k-1）权值系数矩阵，维数3*nh
Delta_uk=[u(1)-u(2);u(1);u(1)-2*u(2)+u(3)];
%增量式PID:Delta_uk=[e(k)-e(k-1);e(k);e(k)+e(k-2)-2*e(k-1)],3*1


%%输入层
O1=[u(6),u(4),u(1)];
%神经网络输入层输入O1=[u(6),u(4),u(1)]=[r(k),y(k),e(k)],1*3


%%隐含层
net2=O1*wi_1';
%隐含层输入,1*nh
O2=tanh(net2);
%隐含层输出,1*nh


%%输出层
net3=O2*wo_1';
%输出层输入,1*3
O3=(1+tanh(net3))/2;
%输出层输出,1*3,对应[kp,ki,kd]


%%梯度下降算法修正网络权系数
uk=(u(7)+O3*Delta_uk)/1.5;
%u(k)值
dydu=sign((u(4)-u(5))*(O3*Delta_uk-u(7)+u(8)));
%%用符号函数取代dydu
delta3=u(1)*dydu.*(((1+tanh(net3))/2).*(1-(1+tanh(net3))/2))'.*Delta_uk;
%delta3,3*1
wo=wo_1+alfa*(wo_1-wo_2)+xite*delta3*O2;
%输出层加权系数,3*nh
delta2=((1-tanh(net2).^2)/2.*(delta3'*wo))';
wi=wi_1+alfa*(wi_1-wi_2)+xite*delta2*O1;
%隐含层加权系数,nh*3
sys = [uk;O3(:);wi(:);wo(:)];