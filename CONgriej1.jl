    using Plots
    using Distances
    using Distributions
    const l=2400 #nro de datos
    const d=2 #dimension
    const hl=.15
    const margen=.2
    const m=20
    ############## DATOS
    U=Uniform(-1,1)
    Ll=collect(-1:0.01:1)
    const r=0.1#paso de la grilla
    L=collect(-1:r:1)
    alph=4
    bdr=(1/2)*sin.(alph*Ll)
    Xl=Array{Float64,2}(undef,d+2,l)
    Xn=Array{Float64,2}(undef,d+2,m)

    function idx(trp,xlpart,h::Float64)::Vector{Int64}
        if length(trp)==0
            return([])
        else
            dist=pairwise(Chebyshev(),xlpart[3:(d+2),:],trp[3:(d+2),:])
            bb=[]
          for i=1:size(trp)[2]
              bb=[bb;findall(vec(dist[:,i].<h))]
        end
           return(bb)
    end
    end


    function idx2(trp,xlpart,h::Float64)::Vector{Int64}
        dist=pairwise(Chebyshev(),xlpart[3:(d+2),:],trp[3:(d+2),:])
        idx=[]
        hh=vec(dist.<h)
        idx=findall(hh)
       return(idx)
    end


function idx3(trp,xlpart,f::Float64)::Vector{Bool}#::Matrix{Float64},xlpart::Matrix{Float64},f::Float64)::Vector{Bool}
    dist=pairwise(Chebyshev(),xlpart[3:(d+2),:],trp[3:(d+2),:])
    idx=[]
    hh=vec(dist.<f)
   return(hh)
end


function remu(I::Vector{Int64},J::Vector{Int64})::Vector{Int64}
    if length(J)==0 return(I)
    else
        I=setdiff(I,J)
        l1=length(I)
        for i=1:l1
        I[i]=I[i]-length(findall(I[i].>=J))
        end
    end
return(I)
end



function etag(trp::Matrix{Float64},x,h::Float64)::Float64
    vot=idx2(x,trp,h)
    return (sum(trp[2,vot])/length(vot))
end



function SS(dinicial::Matrix{Float64},xlpart::Matrix{Float64},Xl::Matrix{Float64},h::Float64)::Matrix{Float64}
    W=size(xlpart)[2]
    trp=dinicial
    indices=unique(idx(trp,xlpart,h))
    P=[]
while W>0
    indices=unique([indices;idx(P,xlpart,h)])
# indices=fastuniq(sort(idx(trp,xlpart))) # puntos de xlpart a menos de h de trp
L=length(indices)
scores=zeros(Float64,L)
xlaux=xlpart[:,indices]
for i=1:L
scores[i]=round(etag(trp,xlaux[:,i],h); digits=2)
end
ceros= scores.<=.5
xlaux[2,ceros].=0
xlaux[2,.!ceros].=1
M1=findmax(scores)
ll=length(scores)
M2=findmax(ones(ll)-scores)
if M1[1]>M2[1]
    MX=[M1[1] 1]
    ind2=findall(@. scores==MX[1])
  #punto que maximiza eta y el cardinal
elseif  M1[1]<M2[1]
    MX=[1-M2[1] 0]
ind2=findall(@. scores==round(MX[1]; digits=2))
else
    unos=findall(@. (scores==M1[1]))
    ind2=unique([unos;findall(@. scores==1-M2[1])])
end
J=size(ind2)[1]
N=zeros(Int64,J)
 # Nro de puntos de xl en los X_i donde se da el maximo
for j=1:J
  N[j]=size(idx2(xlaux[:,ind2[j]],Xl,h))[1]
end
mxcard=findmax(N)
mxs=findall(N.==mxcard[1])
I=ind2[mxs]
P=xlaux[:,I]
trp=[trp P]
u=indices[I]
rr=size(xlpart)[2]
xlpart=xlpart[:,setdiff(1:rr,u)]
indices=remu(indices,u)
W=size(xlpart)[2]
end
return(trp)
end




nrorep=1
    aa=zeros(nrorep)
    res=zeros(nrorep)
for s=1:nrorep
    xx=[rand(U,5000) rand(U,5000)]
    unos=xx[:,2].> (1/2)*sin.(alph*xx[:,1])
    ceros=.!unos
    aux1=(1/2)*sin.(alph*xx[unos,1]).+margen
    aux2=(1/2)*sin.(alph*xx[ceros,1]).-margen
    au=findall(unos)
    au2=findall(ceros)
    ii1=au[findall(xx[unos,2] .>aux1)]
    ii0=au2[findall(xx[ceros,2].<aux2)]
    enmar1=setdiff(au,ii1)
    enmar0=setdiff(au2,ii0)
    bn1=Binomial(l,1/8)
    adent=rand(bn1,1)
    afuera=l-adent[1]
    afun=rand(Binomial(afuera,1/2))
    adun=rand(Binomial(adent[1],1/2))
    xlunosafu=[ones(afun)';ones(afun)';(xx[ii1,:][1:afun,:])']
    xlcerosafu=[zeros(afuera-afun)';zeros(afuera-afun)';(xx[ii0,:][1:(afuera-afun),:])']
    xlunosadent=[ones(adun)';ones(adun)';(xx[enmar1,:][1:adun,:])']
    xlcerosadent=[zeros(adent[1]-adun[1])';zeros(adent[1]-adun[1])';(xx[enmar0,:][1:(adent[1]-adun[1]),:])']
    Xl=[xlunosafu xlcerosafu xlunosadent xlcerosadent]
    yy=[rand(U,5000) rand(U,5000)]
    unos=yy[:,2].> (1/2)*sin.(alph*yy[:,1])
    ceros=.!unos
    bn2=Binomial(m,1/2)
    uns=rand(bn2,1)
    xnunos=yy[unos,:][1:uns[1],:]
    xnceros=yy[ceros,:][1:(m-uns[1]),:]
    xnunos=[ones(uns[1])';ones(uns[1])';xnunos']
    xnceros=[zeros(m-uns[1])';zeros(m-uns[1])';xnceros']
    global Xn=[xnunos xnceros]

test=zeros(0,2)
for i in L
    for j in L
#    global test
    test=[test;[i j]]
end
end


mm=size(test)[1]
test=[zeros(mm) zeros(mm) test]'
M=size(test)[2]
ee=zeros(M)
for i=1:M
    ee[i]=length(idx2(test[:,i],Xl,r/2))
end
test=test[:,findall(@. ee>0)]
global ZZ=SS(Xn,test,Xl,hl)
MM=size(ZZ)[2]
for i=1:MM
    ff=idx3(ZZ[:,i],Xl,r/2)
    global Xl[2,ff].=ZZ[2,i]
end
res[s]=sum(abs.(Xl[1,:]-Xl[2,:]))/l
end







unos=findall(@. Xl[2,:]==1)
ceros=findall(@. Xl[2,:]==0)
plot(Ll,bdr,xlim=(-1,1),ylim=(-1,1),legend=false,axis=false,linewidth=2)
scatter!(Xl[3,unos],Xl[4,unos],marker=[:star5],markersize=1.5,color="red")
scatter!(Xl[3,ceros],Xl[4,ceros],markersize=1.5,color="black")
scatter!(Xn[3,1:m],Xn[4,1:m],markersize=2.2,marker=[:rect],color="yellow")









unos=findall(@. trp[2,:]==1)
ceros=findall(@. trp[2,:]==0)
plot(Ll,bdr,xlim=(-1,1),ylim=(-1,1),legend=false,axis=false,linewidth=2)
scatter!(trp[3,unos],trp[4,unos],marker=[:star5],markersize=1.5,color="red")
scatter!(trp[3,ceros],trp[4,ceros],markersize=1.5,color="black")
scatter!(trp[3,1:m],trp[4,1:m],markersize=2.2,marker=[:rect],color="yellow")


savefig("plot.png")





plot()
unos=findall(@. xlpart[1,:]==1)
ceros=findall(@. xlpart[1,:]==0)
scatter!(xlpart[3,unos],xlpart[4,unos],marker=[:star5],markersize=2,lab="1",xlim=(-1.2,3.3),axis=false)
scatter!(xlpart[3,ceros],xlpart[4,ceros],markersize=2,lab="0")
scatter!(Xn[3,1:2],Xn[4,1:2],marker=[:rect],lab="Training")


savefig("plot.png")





unos=findall(@. B==1)
ceros=findall(@. B==0)
scatter(Xl[3,unos],Xl[4,unos],marker=[:star5],markersize=2,lab="1",xlim=(-1.2,3.3),axis=false)
scatter!(Xl[3,ceros],Xl[4,ceros],markersize=2,lab="0")
scatter!(Xn[3,1:2],Xn[4,1:2],marker=[:rect],lab="Training")
savefig("plot.png")
