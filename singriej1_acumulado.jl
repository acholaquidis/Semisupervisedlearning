
using Plots; pyplot()
using Distances
using Distributions
using LIBSVM
using Random
const d=2 #dimension

const margen=.2
const m=20
nrorep=2
U=Uniform(-1,1)
const r=0.1#paso de la grilla
L=collect(-1:r:1)
Ll=collect(-1:0.01:1)
alph=4
bdr=(1/2)*sin.(alph*Ll)
test=zeros(0,2)
for i in L
    for j in L
    global test
    test=[test;[i j]]
end
end
nmeth=6 # SSL 2h , knncv , SVM, SSLgri




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
dts=pairwise(SqEuclidean(),donde[3:(d+2),:],datos[3:(d+2),:],dims=2)
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
    Xnaux=copy(Xn)
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
    trp=copy(dinicial)
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


function idx3(trp,xlpart,f::Float64)::Vector{Bool}#::Matrix{Float64},xlpart::Matrix{Float64},f::Float64)::Vector{Bool}
    dist=pairwise(Chebyshev(),xlpart[3:(d+2),:],trp[3:(d+2),:])
    idx=[]
    hh=vec(dist.<f)
   return(hh)
end


function SSG(Xn::Matrix{Float64},Xl::Matrix{Float64},test::Matrix{Float64},hl::Float64)::Matrix{Float64}
    mm=size(test)[1]
    test=[zeros(mm) zeros(mm) test]'
    M=size(test)[2]
    ee=zeros(M)
    for i=1:M
    ee[i]=length(idx2(test[:,i],Xl,r/2))
    end
    test=test[:,findall(@. ee>0)]
    ZZ=SS(Xn,test,Xl,hl)
    SSgri=copy(Xl)
    MM=size(ZZ)[2]
    for i=1:MM
        ff=idx3(ZZ[:,i],Xl,r/2)
        SSgri[2,ff].=ZZ[2,i]
    end
    return SSgri
end



function todo(stepl,nrorep,cuantas,test)
    res=Array{Float64,2}(undef,cuantas,3*nrorep)
    times=Array{Float64,2}(undef,cuantas,3*nrorep)
for s=1:nrorep
    Xl=Array{Float64,2}(undef,d+2,0)
    Xn=Array{Float64,2}(undef,d+2,0)
    yy=[rand(U,8000) rand(U,8000)]
    unos=yy[:,2].> (1/2)*sin.(alph*yy[:,1])
    ceros=.!unos
    bn2=Binomial(m,1/2)
    uns=rand(bn2,1)
    xnunos=yy[unos,:][1:uns[1],:]
    xnceros=yy[ceros,:][1:(m-uns[1]),:]
    xnunos=[ones(uns[1])';ones(uns[1])';xnunos']
    xnceros=[zeros(m-uns[1])';zeros(m-uns[1])';xnceros']
    Xn=[xnunos xnceros]
    kcv=cvloo(Xn)
    xx=[rand(U,8000) rand(U,8000)]
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
    bn1=Binomial(stepl*cuantas,1/8)
    adent=rand(bn1,1)
    afuera=stepl*cuantas-adent[1]
    afun=rand(Binomial(afuera,1/2))
    adun=rand(Binomial(adent[1],1/2))
    xlunosafu=[ones(afun)';ones(afun)';(xx[ii1,:][1:afun,:])']
    xlcerosafu=[zeros(afuera-afun)';zeros(afuera-afun)';(xx[ii0,:][1:(afuera-afun),:])']
    xlunosadent=[ones(adun)';ones(adun)';(xx[enmar1,:][1:adun,:])']
    xlcerosadent=[zeros(adent[1]-adun[1])';zeros(adent[1]-adun[1])';(xx[enmar0,:][1:(adent[1]-adun[1]),:])']
    Xlauxi=[xlunosafu xlcerosafu xlunosadent xlcerosadent]
    Xlauxi=Xlauxi[:,shuffle(1:stepl*cuantas)]
for f=1:cuantas
    l=stepl*f
    hl1=0.15
    #hl1= 1.7*(log(l)/l)^(1/4)
    #hl2= hl1 #para ssl grilla
    #hl3= hl1*0.7
    Xl=Xlauxi[1:end,1:l]

    model=svmtrain(Xn[3:end,:],Xn[1,:])
    times[f,3s]= @elapsed predi=svmpredict(model,Xl[3:end,:])[1]
    Xlsvm=copy(Xl)
    Xlsvm[2,:]=predi
    res[f,3s]=sum(abs.(predi-Xl[1,:]))/l    ## ERROR SVM

    times[f,3s-2]= @elapsed trp1=SS(Xn,Xl,Xl,hl1)
    ll1=size(trp1)[2]-m
    res[f,3s-2]=sum(abs.(trp1[1,:]-trp1[2,:]))/ll1 ## ERROR SSL

    #times[f,5s-3]= @elapsed trp2=SSG(Xn,Xl,test,hl2)
    #ll2=size(trp2)[2]-m
    #res[f,5s-3]=sum(abs.(trp2[1,:]-trp2[2,:]))/ll2  ## ERROR SSL-GRILLA

    #times[f,5s-2]= @elapsed  trp3=SS(Xn,Xl,Xl,hl3)
    #ll3=size(trp3)[2]-m
    #res[f,5s-2]=sum(abs.(trp3[1,:]-trp3[2,:]))/ll3

    times[f,3s-1]= @elapsed  Xlaux=knn(Xn,Xl,kcv)
    res[f,3s-1]=sum(abs.(Xlaux[2,:]-Xl[1,:]))/l  ## ERROR knn
end
print(s)
end
return [res,times]
end

cuantas=1
stepl=2400
repet=500
todos=todo(stepl,repet,cuantas,test)
tiemptot=todos[2]




gr1=collect(1:5:(5*repet))
gr2=collect(2:5:(5*repet))
gr3=collect(3:5:(5*repet))
gr4=collect(4:5:(5*repet))
gr5=collect(5:5:(5*repet))


meanT1=(1/repet)*sum(tiemptot[:,gr1],dims=2)
meanSSLG=(1/repet)*sum(tiemptot[:,gr2],dims=2)
meanT3=(1/repet)*sum(tiemptot[:,gr3],dims=2)
meanknn=(1/repet)*sum(tiemptot[:,gr4],dims=2)
meanTSV=(1/repet)*sum(tiemptot[:,gr5],dims=2)


mn1hl1=(1/repet)*sum(todos[:,gr1],dims=2)
mn2hl2=(1/repet)*sum(todos[:,gr2],dims=2)
mn3hl3=(1/repet)*sum(todos[:,gr3],dims=2)
mn4knn=(1/repet)*sum(todos[:,gr4],dims=2)
mn5SVM=(1/repet)*sum(todos[:,gr5],dims=2)

md1=median(todos[:,gr1],dims=2)
md2=median(todos[:,gr2],dims=2)
md3=median(todos[:,gr3],dims=2)
md4=median(todos[:,gr4],dims=2)
md5=median(todos[:,gr5],dims=2)




scatter(md1,label="hl1")
#scatter!(md2,label="hl2")
scatter!(md3,label="hl3")
scatter!(md4,label="knn")
scatter!(md5,label="SVM")


md1=quantile(todos[:,gr1],0.75)




md1=mapslices(x -> quantile(x, 0.75),todos[:,gr1],dims=2)
#md2=mapslices(x -> quantile(x, 0.75),todos[:,gr2],dims=2)
md3=mapslices(x -> quantile(x, 0.75),todos[:,gr3],dims=2)
md4=mapslices(x -> quantile(x, 0.75),todos[:,gr4],dims=2)
md5=mapslices(x -> quantile(x, 0.75),todos[:,gr5],dims=2)




plot(mn1hl1,label="hl1")
#scatter!(mn2hl2,label="hl2")
plot!(mn3hl3,label="hl3")
plot!(mn4knn,label="knn")
plot!(mn5SVM,label="SVM")




md1=findmin(todos[:,gr1],dims=2)[1]
md2=findmin(todos[:,gr2],dims=2)[1]
md3=findmin(todos[:,gr3],dims=2)[1]
md4=findmin(todos[:,gr4],dims=2)[1]
md5=findmin(todos[:,gr5],dims=2)[1]



writedlm("todo505060.csv",gg)








####################  PLOT DE KNN #################################
unos=findall(@. Xlaux[2,:]==1)
ceros=findall(@. Xlaux[2,:]==0)
plot(Ll,bdr,lab="(1/2)sin(4x)",xlim=(-1,1),ylim=(-1,1),legend=false,axis=false,linewidth=2)
scatter!(Xlaux[3,unos],Xlaux[4,unos],marker=[:star5],markersize=1.5,markercolor=[:red])
scatter!(Xlaux[3,ceros],Xlaux[4,ceros],markersize=1.5,color="black")
scatter!(Xn[3,1:m],Xn[4,1:m],markersize=2.2,marker=[:rect],color="yellow")

savefig("knn.png")




####################  PLOT DE ssl #################################
unos=findall(@. trp1[2,:]==1)
ceros=findall(@. trp1[2,:]==0)
plot(Ll,bdr,lab="(1/2)sin(4x)",xlim=(-1,1),ylim=(-1,1),legend=false,axis=false,linewidth=2)
scatter!(trp1[3,unos],trp1[4,unos],marker=[:star5],markersize=1.5,markercolor=[:red])
scatter!(trp1[3,ceros],trp1[4,ceros],markersize=1.5,color="black")
scatter!(Xn[3,1:m],Xn[4,1:m],markersize=2.2,marker=[:rect],color="yellow")

savefig("ssl.png")



####################  PLOT DE ssl gri #################################
unos=findall(@. trp2[2,:]==1)
ceros=findall(@. trp2[2,:]==0)
plot(Ll,bdr,lab="(1/2)sin(4x)",xlim=(-1,1),ylim=(-1,1),legend=false,axis=false,linewidth=2)
scatter!(trp2[3,unos],trp2[4,unos],marker=[:star5],markersize=1.5,markercolor=[:red])
scatter!(trp2[3,ceros],trp2[4,ceros],markersize=1.5,color="black")
scatter!(Xn[3,1:m],Xn[4,1:m],markersize=2.2,marker=[:rect],color="yellow")

savefig("sslgri.png")



################################ plot SVM
unos=findall(@. Xlsvm[2,:]==1)
ceros=findall(@. Xlsvm[2,:]==0)
plot(Ll,bdr,lab="(1/2)sin(4x)",xlim=(-1,1),ylim=(-1,1),legend=false,axis=false,linewidth=2)
scatter!(Xlsvm[3,unos],Xlsvm[4,unos],marker=[:star5],markersize=1.5,markercolor=[:red])
scatter!(Xlsvm[3,ceros],Xlsvm[4,ceros],markersize=1.5,color="black")
scatter!(Xn[3,1:m],Xn[4,1:m],markersize=2.2,marker=[:rect],color="yellow")

savefig("svm.png")




#plot(L,bdr,lab="(1/2)sin(4x)",xlim=(-1.1,1.1),ylim=(-2,1.1),axis=false,linewidth=2,legend=:bottom)
#scatter!(trp[3,unos],trp[4,unos],marker=[:star5],markersize=1.5,lab="1",color="red")
#scatter!(trp[3,ceros],trp[4,ceros],markersize=1.5,lab="0",color="black")
#scatter!(trp[3,1:m],trp[4,1:m],markersize=2.2,marker=[:rect],lab="Training",color="yellow")

savefig("semisup_2000_02_15.png")


knn=5


unos=findall(@. Xl[1,:]==1)
ceros=findall(@. Xl[1,:]==0)
plot(Ll,bdr,lab="(1/2)sin(4x)",xlim=(-1,1),ylim=(-1,1),legend=false,axis=false,linewidth=2)
scatter!(Xl[3,unos],Xl[4,unos],marker=[:star5],markersize=1.5,color="red")
scatter!(Xl[3,ceros],Xl[4,ceros],markersize=1.5,color="black")
scatter!(Xn[3,1:m],Xn[4,1:m],markersize=2.2,marker=[:rect],color="yellow")






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




plot(md1,label="hl1")
#scatter!(md2,label="hl2")
plot!(md3,label="hl3")
plot!(md4,label="knn")
plot!(md5,label="SVM")
