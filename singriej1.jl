using Plots; pyplot()

using Distances
using Distributions
using LIBSVM
const d=2 #dimension

const margen=.2
const m=20
nrorep=1000
U=Uniform(-1,1)
Ll=collect(-1:0.01:1)
alph=4
bdr=(1/2)*sin.(alph*Ll)



function idx(trp,xlpart,h::Float64)::Vector{Int64}
                if length(trp)>0
                    dist=pairwise(SqEuclidean(),xlpart[3:(d+2),:],trp[3:(d+2),:],dims=2)
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
    dist=pairwise(SqEuclidean(),xlpart[3:(d+2),:],trp[3:(d+2),:],dims=2)
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

function knn(datos,donde,k)
dts=pairwise(Euclidean(),donde[3:(d+2),:],datos[3:(d+2),:],dims=2)
ll=size(donde)[2]
 for i=1:ll
    vot=sum(datos[1,sortperm(dts[i,:])[1:k]])/k
    if vot>1/2
        donde[2,i]=1
    else
        donde[2,i]=0
    end
end
return(donde)
end




function cvloo(Xn)
    Xnaux=Xn
inn=size(Xn)[2] #tamaño del a muestra Xn
gri=collect(1:2:17)
cv=[]#para el valor de CV en la grilla
for k in gri
        for i=1:inn
        dtsUaux=setdiff(1:inn,i) ## para sacar el indice i
        v=reshape(Xn[:,i],4,1)
        Xnaux[:,i]=knn(Xnaux[:,dtsUaux],v,k) #kernel con la muestra sin el i
    end
    cv=[cv;sum(abs.(Xnaux[2,:]-Xnaux[1,:]))/inn]
  end
return gri[findmin(cv)[2]] #valor de la grilla donde se da el minimo
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


#grill=collect(2400:1:2400)
#nrogri=length(grill)
#meanl=Array{Float64,2}(undef,nrogri,3)   #3hs knn svm
l=2400
#for l in grill
global Xl=Array{Float64,2}(undef,d+2,l)
global Xn=Array{Float64,2}(undef,d+2,m)
hl1= 0.15
global res=Array{Float64,2}(undef,nrorep,3)  # 3hs knn svm
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
    #model=svmtrain(Xn[3:end,:],Xn[1,:])
    #predi=svmpredict(model,Xl[3:end,:])[1]
    #res[s,5]=sum(abs.(predi-Xl[1,:]))/l
    #global trp1=SS(Xn,Xl,Xl,hl1)
    #ll1=size(trp1)[2]-size(Xn)[2]
    #res[s,1]=sum(abs.(trp1[1,:]-trp1[2,:]))/ll1
#trp2=SS(Xn,Xl,Xl,hl2)
#ll2=size(trp2)[2]-size(Xn)[2]
#res[s,2]=sum(abs.(trp2[1,:]-trp2[2,:]))/ll2
#trp3=SS(Xn,Xl,Xl,hl3)
#ll3=size(trp3)[2]-size(Xn)[2]
#res[s,3]=sum(abs.(trp3[1,:]-trp3[2,:]))/ll3
global Xl=knn(Xn,Xl,1)
res[s,1]=sum(abs.(Xl[2,:]-Xl[1,:]))/l
global Xl=knn(Xn,Xl,3)
res[s,2]=sum(abs.(Xl[2,:]-Xl[1,:]))/l
global Xl=knn(Xn,Xl,5)
res[s,3]=sum(abs.(Xl[2,:]-Xl[1,:]))/l
end
#meanl[grill.==l,:]=sum(res,dims=1)/nrorep
#end












unos=findall(@. trp1[2,:]==1)
ceros=findall(@. trp1[2,:]==0)
plot(Ll,bdr,lab="(1/2)sin(4x)",xlim=(-1,1),ylim=(-1,1),legend=false,axis=false,linewidth=2)
scatter!(trp1[3,unos],trp1[4,unos],marker=[:star5],markersize=1.5,markercolor=[:red])
scatter!(trp1[3,ceros],trp1[4,ceros],markersize=1.5,color="black")
scatter!(trp1[3,1:m],trp1[4,1:m],markersize=2.2,marker=[:rect],color="yellow")



#plot(L,bdr,lab="(1/2)sin(4x)",xlim=(-1.1,1.1),ylim=(-2,1.1),axis=false,linewidth=2,legend=:bottom)
#scatter!(trp[3,unos],trp[4,unos],marker=[:star5],markersize=1.5,lab="1",color="red")
#scatter!(trp[3,ceros],trp[4,ceros],markersize=1.5,lab="0",color="black")
#scatter!(trp[3,1:m],trp[4,1:m],markersize=2.2,marker=[:rect],lab="Training",color="yellow")

savefig("ssl.png")


knn=5


unos=findall(@. Xl[2,:]==1)
ceros=findall(@. Xl[2,:]==0)
plot(Ll,bdr,lab="(1/2)sin(4x)",xlim=(-1,1),ylim=(-1,1),legend=false,axis=false,linewidth=2)
scatter!(Xl[3,unos],Xl[4,unos],marker=[:star5],markersize=1.5,color="red")
scatter!(Xl[3,ceros],Xl[4,ceros],markersize=1.5,color="black")
scatter!(Xn[3,1:m],Xn[4,1:m],markersize=2.2,marker=[:rect],color="yellow")


savefig("knn3.png")





plot()
unos=findall(@. xlpart[1,:]==1)
ceros=findall(@. xlpart[1,:]==0)
scatter!(xlpart[3,unos],xlpart[4,unos],marker=[:star5],markersize=2,lab="1",xlim=(-1.2,3.3),axis=false)
scatter!(xlpart[3,ceros],xlpart[4,ceros],markersize=2,lab="0")
scatter!(Xn[3,1:2],Xn[4,1:2],marker=[:rect],lab="Training")


savefig("knn5.png")





unos=findall(@. B==1)
ceros=findall(@. B==0)
scatter(Xl[3,unos],Xl[4,unos],marker=[:star5],markersize=2,lab="1",xlim=(-1.2,3.3),axis=false)
scatter!(Xl[3,ceros],Xl[4,ceros],markersize=2,lab="0")
scatter!(Xn[3,1:2],Xn[4,1:2],marker=[:rect],lab="Training")
savefig("plot.png")
