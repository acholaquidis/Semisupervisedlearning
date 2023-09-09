    using Plots
    using Distances
    using Distributions
    l=2000 #nro de datos
    n=2
    d=2 #dimension
    p=1/2
    v=1.5 #para traslación
    hl=.4
    ############## DATOS
    U1=Normal(0,.6)
    Ber=Bernoulli(p)
    B=rand(Ber,l)

    Xl=Array{Float64,2}(undef,d+2,l)
    Xn=Array{Float64,2}(undef,d+2,2)



function idx(trp,xlpart,h::Float64)::Vector{Int64}
                if length(trp)>0
                    dist=pairwise(SqEuclidean(),xlpart[3:(d+2),:],trp[3:(d+2),:])
                    bb=[]
                  for i=1:size(trp)[2]
                      bb=[bb;findall(dist[:,i].<h^2)]
                end
                   return(bb)
               else
                    return([])
            end
        end



function idx2(trp,xlpart::Matrix{Float64},h::Float64)::Vector{Int64}
    dist=pairwise(SqEuclidean(),xlpart[3:(d+2),:],trp[3:(d+2),:])
   return(findall(vec(dist.<h^2)))
end


function etag(trp::Matrix{Float64},x::Vector{Float64},h::Float64)::Float64
    vot=idx2(x,trp,h)
    return (sum(trp[2,vot])/length(vot))
end

function remu(I::Vector{Int64},J::Vector{Int64})::Vector{Int64}
        I=setdiff(I,J)
        for i=1:length(I)
        I[i]=I[i]-length(findall(I[i].>=J))
        end
return(I)
end


function SS(dinicial::Matrix{Float64},xlpart::Matrix{Float64},Xl::Matrix{Float64},h::Float64)::Matrix{Float64}
    W=size(xlpart)[2]
    trp=dinicial
    indices=idx(trp,xlpart,h)
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
ceros= scores.<.5
xlaux[2,ceros].=0
xlaux[2,.!ceros].=1
M1=findmax(scores)
ll=length(scores)
M2=findmax(ones(ll)-scores)
if M1[1]>M2[1]
    MX=[M1[1] 1]
    ind2=findall(@. scores==round(MX[1];digits=2))
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
xlpart=xlpart[:,setdiff(1:(size(xlpart)[2]),u)]
indices=remu(indices,u)
W=size(xlpart)[2]
end
return(trp)
end


nrorep=50
    aa=zeros(nrorep)
    res=zeros(nrorep)
for s=1:nrorep
    #global l
    #global Xn
 D1=rand(U1,d,5*l)
 D2=rand(U1,d,5*l)
N1=vec(sqrt.(sum((D1').^2,dims=2)))
D1=D1[:,N1.<1.5]
N1=vec(sqrt.(sum((D2').^2,dims=2)))
D2=D2[:,N1.<1.5]
D2=D2.+v
Xn[3:(d+2),1]=rand(Normal(0,1/10),d,1)
Xn[1:2,1].=1
Xn[1:2,2].=0
Xn[3:(d+2),2]=rand(Normal(0,1/10),d,1).+v
Xl[1,:]=B
for i=1:l
Xl[3:(d+2),i]= B[i]*D1[:,i+1]+(1-B[i])*D2[:,i+1]
end
aa[s]= @elapsed trp=SS(Xn,Xl,Xl,hl)
ll=size(trp)[2]-size(Xn)[2]
res[s]=sum(abs.(trp[1,:]-trp[2,:]))/ll
end








unos=findall(@. trp[2,:]==1)
ceros=findall(@. trp[2,:]==0)
scatter(trp[3,unos],trp[4,unos],marker=[:star5],markersize=2,lab="1",xlim=(-1.2,v+2),axis=false)
scatter!(trp[3,ceros],trp[4,ceros],markersize=2,lab="0")
scatter!(trp[3,1:2],trp[4,1:2],marker=[:rect],lab="Training")
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
