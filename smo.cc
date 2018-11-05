#include <bits/stdc++.h>
#include <stdlib.h>
#define pb(x) push_back(x)
#define clr(x,y) memset(x,y,sizeof x)
#define ll long long
#define ull unsigned long long
#define pii pair<int,int>
#define pi acos(-1)
#define mkii(x,y)   makepair<int,int>(x,y) ;
#define vd vector<double>
#define vdd vector<vector<double> >
#define sgn(x) (x>0)?1:-1

using namespace std;

const double eps = 1e-10;
const double TOLERANCE = 0.001;
vdd x,test_x;
double sig = 1;
int n , dim ,test_n;
vd y,test_y;
vd alpha,error_cache;

double b,C;
void init() // input data, and clean memory
{
    //freopen("liver-disorders_scale.txt","r",stdin);
    FILE *f = fopen("liver-disorders_scale.txt","r");
    C = 1;
    dim = 5;
    n = 0;
    test_n = 0;
    int tmp  , label;
    double _x;
    int pre = 0;
    x.clear();
    y.clear();
    test_x.clear();
    test_y.clear();
    alpha.clear();
    b = 0;
    while ( fscanf(f,"%d",&label)!=EOF )
    {
        y.pb(label==0?-1:1);
        x.pb(vector<double>());
        pre = 0;
        for (int i = 0 ; i < dim ; i++)
            {
                fscanf(f,"%d:%lf",&tmp, &_x);
                for (int j = pre+1 ; j < tmp ; j++)
                    x[n].pb(_x) , i++;
                pre = tmp;
                x[n].pb(_x);
            }
        n++;
    }
    pre = 0;
    FILE *f_test = fopen("test_data","r");
     while ( fscanf(f_test,"%d",&label)!=EOF )
    {
        test_y.pb(label==0?-1:1);
        test_x.pb(vector<double>());
        pre = 0;
        for (int i = 0 ; i < dim ; i++)
            {
                fscanf(f_test,"%d:%lf",&tmp, &_x);
                for (int j = pre+1 ; j < tmp ; j++)
                    test_x[test_n].pb(_x) , i++;
                pre = tmp;
                test_x[test_n].pb(_x);
            }
        test_n++;
    }
}
double dot ( vector<double> a , vector<double> b) // dot product of vector a and b
{
    double ret = 0;
    for (int i = 0 ; i < dim ; i++)
        ret += a[i] * b[i];
    return ret;
}

double linear_kernel(vector<double> a, vector<double> b) { return dot(a,b); } // linear kernel
double RBF_kernel(vector<double> a, vector<double> b)
{ //rbf kernel
    double k = dot(a,b);
    k *= -2.0;
    k += dot(a,a) + dot(b,b);
    k /= -(2.0 * sig * sig);
    k = exp(k);
    return k;
}

double K(vector<double> a ,vector<double> b)
{
    return RBF_kernel(a,b);
    //return linear_kernel(a,b);
}
double f(vector<double> X) //return object function
{
    double sum = -b;
    for (int i = 0 ; i < n ; i++)
        sum += alpha[i] * y[i] * K(x[i],X);
    return sum;
}
//use error cache to speed up SMO
int take_step(int _i,int _j) //implement SMO to update alpha1 and alpha2
{
    if ( _i == _j ) return 0;
    double K11 = K(x[_i],x[_i]),K22=K(x[_j],x[_j]),K12=K(x[_i],x[_j]),K21=K(x[_j],x[_i]);
    double y1 = y[_i] , y2 = y[_j];
    double s = y1 * y2;
    double E1 , E2;
    if (alpha[_i] > 0 && alpha[_i] < C) E1 = error_cache[_i]; else E1 = f(x[_i]) - y1;
    if (alpha[_j] > 0 && alpha[_j] < C) E2 = error_cache[_j]; else E2 = f(x[_j]) - y2;
    double _E1 = f(x[_i]) - y1;
    double _E2 = f(x[_j]) - y2;


    double eta = K11 + K22 - 2* K12;
    double alpha1 = alpha[_i];
    double alpha2 = alpha[_j];
    double a1 , a2;
    double L,H;
    if (s < 0) { L = max(0.,alpha2 -alpha1 ),H=min(C,C+alpha2-alpha1); }
    else { L=max(0.,alpha2+alpha1-C),H=min(C,alpha1+alpha2); }

    //cout << L << " " << H << endl;
    if ( H - L < eps )  return 0;
    if (eta > 0)
    {
        a2 =alpha2 + y2 *(E1-E2)/eta;
        if ( a2 < L) a2 = L;
        else if ( a2 > H) a2 = H;
    }
    else //i find eta < 0 will not happen
    {
//        double f1 = y1*(E1 + b) - alpha1 * K11 - s * alpha2 * K12;
//        double f2 = y2*(E2 + b) - s * alpha1 * K21 - alpha2 * K22;
//        double L1 = alpha1 + s * (alpha2-L);
//        double H1 = alpha1 + s * (alpha2-H);
//        double Lobj = L1 * f1 + L * f2 + 0.5 * L1 * L1 * K11 + 0.5 * L * L * K22 + s*L*L1*K12;
//        double Hobj = H1 * f1 + H * f2 + 0.5 * H1 * H1 * K11 + 0.5 * H * H * K22 + s*H*H1*K12;
//        if ( Lobj < Hobj - eps)
//            a2 = L;
//        else if ( Lobj > Hobj + eps)
//            a2 = H;
//        else
//            a2 = alpha2;
//
//        cout << "???" << endl;
          return 0;
    }

    if (fabs(a2 - alpha2) < eps) return 0;

    a1 = alpha1 + y1 * y2 * (alpha2 - a2);
    double b1_new,b2_new,b_new;
    b1_new = E1 + y[_i]*K11*(a1 - alpha1) + y[_j]*K21*(a2 - alpha2) + b;
    b2_new = E2 + y[_i]*K12*(a1 - alpha1) + y[_j]*K22*(a2 - alpha2) + b;
    if (  ( a2 < 0 || a2 > C ) && (a1 >=0 && a1 <=C) ) b_new = b1_new;
    else if ( ( a1 >= 0 && a1 <=C ) && ( a2 < 0 || a2 > C) ) b_new = b2_new;
    else b_new =  ( b1_new + b2_new) / 2;

    alpha[_i] = a1; //update alpha
    alpha[_j] = a2;


    double t1 = y1 * (a1 - alpha1); // update error cache
    double t2 = y2 * (a2 - alpha2);

    for (int i = 0 ; i < n ; i++)
        {
        if (alpha[i] > 0 && alpha[i] < C && i != _i && i != _j)
         error_cache[i] += t1 * K(x[_i], x[i]) + t2 * K(x[_j], x[i]) + b - b_new;
        }
        b = b_new;
    error_cache[_i] = f(x[_i]) - y[_i];
    error_cache[_j] = f(x[_j]) - y[_j];
    return 1;
}
int examine_example(int i1) { //example alpha broke KKT condition

    double y1 = 0.0;
    double alpha1 = 0.0;
    double e1 = 0.0;
    double r1 = 0.0;
    y1 = y[i1];
    alpha1 = alpha[i1];
    if (alpha1 > 0 && alpha1 < C)
        e1 = error_cache[i1];
    else
        e1 = f(x[i1]) - y1;
    r1 = y1 * e1;
    if ((r1 < -TOLERANCE && alpha1 < C) || (r1 > TOLERANCE && alpha1 > 0))
    {
        int k0 = 0;
        int k = 0;
        int i2 = -1;
        double tmax = 0.0 , temp;
        for (i2 = -1, tmax = 0, k = 0; k < n; k++)
            if (alpha[k] > 0 && alpha[k] < C )
                if ( (temp = fabs(e1 - error_cache[k])) > tmax)  tmax = temp , i2 = k;
        if (i2 >= 0) { if  (take_step(i1, i2)) return 1; }

        for (k0 = rand() % n, k = k0 ; k < n + k0; k++)
            { i2 = k % n;  if (alpha[i2] > 0 && alpha[i2] < C)  if (take_step(i1, i2)) return 1; }
        for (k0 = rand() % n, k = k0 ; k < n + k0; k++)
            { i2 = k % n; if (take_step(i1, i2)) return 1; }
    }
    return 0;
}
void train() //trainmodel
{
    alpha.resize(n, 0);
    b = 0.0;
    error_cache.resize(n, 0);
    int num_changed = 0;
    for (int k = 0; k < n; k++)
        num_changed += examine_example(k);
    int maxGen = 233,_=0;
    while (num_changed > 0 && _++ < maxGen)
    {
        num_changed = 0;
        for (int k = 0; k < n; k++)
            if (alpha[k] != 0 && alpha[k] != C)
            num_changed += examine_example(k);
    }

}

void test() // test result
{
    int error = 0;
    int TP = 0 , TN = 0 , FP = 0 , FN = 0;
    for (int i = 0 ; i < test_n ; i++)
        {
            int predict_label = sgn(f(test_x[i]));
            if (predict_label > 0 && test_y[i] > 0) TP ++;
            if (predict_label > 0 && test_y[i] < 0) TN ++;
            if (predict_label < 0 && test_y[i] > 0) FP ++;
            if (predict_label < 0 && test_y[i] < 0) FN ++;
        }

    for (int i = 0 ; i < n ; i++)
        printf("%g%c",alpha[i],i==n-1?'\n':' ');
    double Acc = (TP + FN) * 1. / (TP + FN + FP + FN);
    cout << " Accuracy = " << Acc << endl;

}
int main()
{
    srand((unsigned)time(NULL));
    init();
    #include <time.h>
    time_t t = clock();
    train();
    test();

    cout << "total time = " << clock() - t << "ms" <<endl;
    return 0;
}
