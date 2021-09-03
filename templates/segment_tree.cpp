int n, t[4*N];
//building the tree
void build(int a[], int v=1, int tl=1, int tr=n) {
    if (tl == tr) {
        t[v] = a[tl];
    } else {
        int tm = (tl + tr) / 2;
        build(a, v*2, tl, tm);
        build(a, v*2+1, tm+1, tr);
        t[v] = t[v*2] + t[v*2+1];
    }
}
//queries
int sum(int l, int r,int v=1, int tl=1, int tr=n ) {
    if (l > r) 
        return 0;
    if (l == tl && r == tr)
        return t[v];
    int tm = (tl + tr) / 2;
    return sum( l, min(r, tm),v*2, tl, tm)
           + sum(max(l, tm+1), r,v*2+1, tm+1, tr );
}
//
void update(int pos, int new_val,int v=1, int tl=1, int tr=n ) {
    if (tl == tr) {
        t[v] = new_val;
    } else {
        int tm = (tl + tr) / 2;
        if (pos <= tm)
            update(pos,new_val,v*2, tl, tm);
        else
            update(pos,new_val,v*2+1, tm+1, tr);
        t[v] = t[v*2] + t[v*2+1];
    }
}

// lazy propagation

void f(int v,int tl,int tr){
  t[v]+=lazy[v];
  if(tl!=tr){
    lazy[v*2]+=lazy[v];
    lazy[v*2+1]+=lazy[v];
  }
  lazy[v]=0;
}
//queries
int sum(int l, int r,int v=1, int tl=0, int tr=2*n+10 ) {
    f(v,tl,tr);
    if (l>tr or r<tl) 
        return 0;
    if (l <= tl and r >= tr)
        return t[v];
    int tm = (tl + tr) / 2;
    return sum( l, r,v*2, tl, tm)
           + sum(l, r,v*2+1, tm+1, tr );
}
//
void update(int l, int r, int new_val,int v=1, int tl=0, int tr=2*n+10 ) {
   f(v,tl,tr);
   if(l>tr or r<tl)return;
   if(l<=tl and r>=tr){
    lazy[v]+=new_val;
    f(v,tl,tr);
    return;
   }
   int tm=tl+tr>>1;
   update(l,r,new_val,v*2,tl,tm);
   update(l,r,new_val,v*2+1,tm+1,tr);
   t[v]=t[v*2]+t[v*2+1];
}