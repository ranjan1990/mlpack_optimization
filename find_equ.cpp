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
        data::Load("traj.txt",data1);
        cout<<" "<<data1.n_cols<<"   "<<data1.n_rows<<endl;
       }

       size_t NumFunctions() const { return 20; }
       arma::mat GetInitialPoint() const { return arma::mat("0.555;0.2203;-0.5269;6.1637"); }
       double Evaluate(const arma::mat& coordinates, const size_t i) const ;
       void Gradient(const arma::mat& coordinates,const size_t i,arma::mat& gradient) const ;
   private:
      arma::mat data1;

};


double MyFunction::Evaluate(const arma::mat& coordinates, const size_t i) const
{
      arma::mat Ri=data1.col(i);

      //Ri.insert_rows(3,1);
     // Ri[3]=1;
      double val=0;
      for(size_t j=0;j<3;j++)
      {
          val=val+ Ri[j]*coordinates[j];

      }
      val=val+ coordinates[3] ; 

      return(val*val);
}

void MyFunction::Gradient(const arma::mat& coordinates, const size_t i, arma::mat& gradient) const
{
  gradient.zeros(4);
  arma::mat Ri=data1.col(i);
      double val=0;
      for(size_t j=0;j<3;j++)
      {
          val=val+ Ri[j]*coordinates[j];

      }
      val=val+ coordinates[3] ; 

  gradient[i]=2*val*Ri[i];

}

main()
{
  
  
  MyFunction f=MyFunction();
  arma::mat gradient; 
 // arma::mat coordinates1 = f.GetInitialPoint();
  arma::mat coordinates2 = f.GetInitialPoint();

//  ParallelSGD<MyFunction> s1(f, 0.00003, 10000000, 1e-9);
  SGD<MyFunction> s2(f, 0.00003, 10000, 1e-9, false);

  //double result1 = s1.Optimize(coordinates1);
  double result2 = s2.Optimize(coordinates2);

   
  for(int i=0;i<50;i++)
  {

    //cout<<f.Evaluate(coordinates1,i)<<endl;
    cout<<f.Evaluate(coordinates2,i)<<endl;
  }
   
//    cout<<coordinates1<<endl;
    cout<<coordinates2<<endl;


}


