#include <RcppArmadillo.h>
#include <RcppTN.h>
#include <math.h>
#include <time.h>
#include <algorithm>

// [[Rcpp::depends(RcppArmadillo)]]
/*// [[Rcpp::depends(RcppTN)]]*/

using namespace Rcpp;
using namespace arma;

inline int randWrapper(const int n) { return floor(unif_rand()*n); }
Rcpp::IntegerVector randomShuffle(Rcpp::IntegerVector a) {
    /* clone a into b to leave a alone */
    Rcpp::IntegerVector b = Rcpp::clone(a);
    std::random_shuffle(b.begin(), b.end(), randWrapper);

    return b;
}

mat rwishart(int const& nu, mat const& V){

  int m = V.n_rows;
  mat T = zeros(m,m);
  mat R = zeros(m,m);
  for(int i = 0; i < m; i++) {
    T(i,i) = sqrt(rchisq(1,nu-i)[0]); 
  }
  
  for(int j = 0; j < m; j++) {  
    for(int i = j+1; i < m; i++) {    
      T(i,j) = rnorm(1)[0]; 
    }}
  
  mat C = trans(T)*chol(V);
  
    return R = trans(C)*C;
}

mat rinvwishart(int const& nu, mat const& V){

  int m = V.n_rows;
  mat T = zeros(m,m);
  mat IR = zeros(m,m);
  for(int i = 0; i < m; i++) {
    T(i,i) = sqrt(rchisq(1,nu-i)[0]);
  }
  
  for(int j = 0; j < m; j++) {  
    for(int i = j+1; i < m; i++) {    
      T(i,j) = rnorm(1)[0]; 
    }}
  
  mat C = trans(T)*chol(V);
  mat CI = solve(trimatu(C),eye(m,m)); 

    return IR = CI*trans(CI);
}

// [[Rcpp::export]]

List BayesH(mat M, mat Y, double nu, mat Omega, double gamma, int nIter){
  int r             = M.n_rows;
  int p             = M.n_cols;
  int nAnim         = Y.n_cols;
  double ttime      = 0;
  clock_t start;
  clock_t end;
  mat Theta         = ones(p, nAnim)/p;
  mat store_Sigma   = zeros(nIter, r);
  mat store_Psi     = zeros(nIter, r);
  cube store_Theta  = zeros(nIter, p, nAnim);
  List Out;
  
  /* For updating ThetaStar */
  vec ThetaStar = zeros(p);
  mat One       = ones <mat> (1, p - 1);
  IntegerVector plants;
  IntegerVector inTheta;
  IntegerVector tmp;
  IntegerVector lastj;
  mat Mstar     = zeros(r, p - 1);
  vec mp        = zeros(r);
  //int estim = 0;
  //int last = 0;
  uvec estim(p - 1);
  uvec last(1);
  vec Var(1);
  vec Mean(1);
  vec Mu = zeros(1);
  

  /*For updating Sigma*/
  mat Sigma = eye(r, r);
  mat Psi   = eye(r, r);
  mat R     = zeros(r, r);
  
  start = clock();
  
  for(int iter = 0; iter < nIter; iter++){
    for(int i = 0; i < nAnim; i++){
      plants                  = seq_len(p);
      tmp                     = randomShuffle(plants);
      lastj                   = tmp(0);
      inTheta                 = Rcpp::setdiff(tmp, lastj);
      estim = inTheta(0) - 1;
      last =  lastj(0) - 1;
      Mstar                   = M.cols(estim);
      mp                      = M.cols(last);
      Var                     = (trans(Mstar - mp*trans(One))*solve(Sigma, Mstar - mp*trans(One)));
      Mean                    = ((1/Var[0])*trans(Mstar - mp*trans(One))*solve(Sigma, Y.col(i) - mp));
      Mu                      = RcppTN::rtn1(Mean(0), sqrt(1/Var[0]), 0.0, 1.0);
      ThetaStar.elem(estim)   = Mu;
      ThetaStar.elem(last)    = 1 - Mu;
      store_Theta(span(iter), span::all, span(i)) = ThetaStar;

      R = R + (Y.col(i) - M*ThetaStar)*trans(Y.col(i) - M*ThetaStar);      
    }
    
    Sigma                 = rinvwishart(nAnim + nu, inv(R + Psi));
    //Sigma                 = rwishart(nAnim + nu, R + Psi)/(pow(nAnim + nu, 2));
    Psi                   = rwishart(nu + gamma, Omega + Sigma);
    store_Sigma.row(iter) = trans(Sigma.diag()); 
    store_Psi.row(iter)   = trans(Psi.diag()); 
    R                     = zeros(r, r);
    
    if(iter % 1000 == 0){
        Rcpp::Rcout << " Iteration " << iter << " out of " << nIter << std::endl;
      }
    
  }
  end = clock();
  ttime = ((double) (end - start)) / CLOCKS_PER_SEC;
  Rcpp::Rcout << nIter << " iterations in " << ttime << " seconds" << std::endl;
  
  Out["Theta"]        = store_Theta;
  Out["Sigma"]        = store_Sigma;
  Out["Psi"]          = store_Psi;
  Out["Elapsed time"] = ttime;
  return(wrap(Out));
}