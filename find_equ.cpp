#include <mlpack/core.hpp>
#include <mlpack/methods/neighbor_search/neighbor_search.hpp>
#include <mlpack/core/optimizers/sgd/sgd.hpp>
#include <mlpack/core/optimizers/parallel_sgd/sgdp.hpp>


using namespace mlpack::optimization;
using namespace mlpack;
using namespace std;
using namespace mlpack::neighbor; // NeighborSearch and NearestNeighborSort
using namespace mlpack::metric; // ManhattanDistance


// 
// (a,b,c,d)= min sum_over_i(a*x_i+b*y_i+c*t_i+d)^2
//number of function = max value of i 



class MyFunction
{
   public:
       MyFunction() {
          data.load("traj.txt");

       }
       size_t NumFunctions() const { return 20; }
       arma::mat GetInitialPoint() const { return arma::mat("0.555;0.2203;-0.5269;6.1637"); }
       double Evaluate(const arma::mat& coordinates, const size_t i) const ;
       void Gradient(const arma::mat& coordinates,const size_t i,arma::mat& gradient) const ;
    private:
      arma::mat data;

};


double MyFunction::Evaluate(const arma::mat& coordinates, const size_t i) const
{
      arma::mat Ri=data.row(i);
      Ri.insert_cols(3,1);
      Ri[3]=1;
      double val=arma::det(coordinates.t()*Ri.t());
      return(val*val);
}

void MyFunction::Gradient(const arma::mat& coordinates, const size_t i, arma::mat& gradient) const
{
  gradient.zeros(4);
  arma::mat Ri=data.row(i);
  Ri.insert_cols(3,1);
  Ri[3]=1;
  double val=arma::det(coordinates.t()*Ri.t());
  gradient[i]=2*val*Ri[i];

}

main()
{
  MyFunction f;
 
  arma::mat gradient; 
  
  SGD<MyFunction> s(f, 0.000003, 5000, 1e-14, false);
  arma::mat coordinates = f.GetInitialPoint();
  double result1 = s.Optimize(coordinates);
 /* 
  for(int i=0;i<50;i++)
  {

    cout<<f.Evaluate(coordinates,i)<<endl;
  }
*/


}


