!********************************************************************************************
      module constants
!**************************************************************************************
!*    This module defines many constants often uesd in our code.
!*    One can use these constants through a command "use constants" in their
!*    own subroutines or functions. 
!************************************************************************************** 
      implicit none
      Double precision  infinity,pi,dtors,sixteen,twopi,zero,one,two,three,four,six,half,&
             half2,mh,hbar,pho_v,plankc,five,dtor,eight 
      parameter(infinity=1.D40,dtors=asin(1.D0)*2.D0/180.D0, &
               sixteen=16.D0, twopi=4.D0*dasin(1.D0), pi = dasin(1.D0)*2.D0) 
      PARAMETER(zero=0.D0, one=1.D0, two=2.D0, three=3.D0, four=4.D0, six=6.D0, half=0.5D0, half2=0.25D0, &
                mh=1.6726231D-24, hbar = 1.0545887D-27, plankc=6.626178D-27, pho_v=2.99792458D10, five=5.D0,&
                dtor=asin(1.D0)*2.D0/180.D0, eight=8.D0)    
!**************************************************************************************
      end module constants
!************************************************************************************** 




!********************************************************************************************
      module rootfind
!********************************************************************************************
!*    This module aims on solving cubic and quartic polynomial equations.
!*    One can use these subroutines root3 and root4 to find roots of cubic and 
!*    quartic equations respectively. 
!********************************************************************************************
      use constants
      implicit none

      contains
!*********************************************************************************************
      subroutine root3(b,c,d,r1,r2,r3,del)
!*******************************************************************************************
!* PURPOSE:   This subroutine aims on solving cubic equations: x^3+b*x^2+c*x+d=0.
!* INPUTS:    b, c, d-----they are the coefficients of the cubic equation. 
!* OUTPUTS:   r1,r2,r3----roots of the equation, with complex number form.
!*            del---------the number of real roots among r1,r2,r3. 
!* ROUTINES CALLED:  sort    
!* This code comes from internet.
!*******************************************************************************************
      implicit none
      Double precision a,b,c,d,p,q,delta,DD,e1,e2,e3,realp,imagep,temp1,temp2,phi,y1,y2,&
             y3,y2r,y2i,u,v
      complex*16 r1,r2,r3
      integer del
 
      a=1.D0 
! Step 1: Calculate p and q --------------------------------------------
      p  = c/a - b*b/a/a/3.D0
      q  = (two*b*b*b/a/a/a - 9.D0*b*c/a/a + 27.D0*d/a) / 27.D0

! Step 2: Calculate DD (discriminant) ----------------------------------
      DD = p*p*p/27.D0 + q*q/4.D0
      !write(*,*)'sssssggggg=',DD 
      !if(dabs(DD).le.1.D-10)DD=0.D0 

! Step 3: Branch to different algorithms based on DD -------------------
      if(DD .lt. 0.D0)then
!       Step 3b:
!       3 real unequal roots -- use the trigonometric formulation
        phi = acos(-q/two/sqrt(abs(p*p*p)/27.D0))
        temp1=two*sqrt(abs(p)/3.D0)
        y1 =  temp1*cos(phi/3.D0)
        y2 = -temp1*cos((phi+pi)/3.D0)
        y3 = -temp1*cos((phi-pi)/3.D0)
      else
!       Step 3a:
!       1 real root & 2 conjugate complex roots OR 3 real roots (some are equal)
        temp1 = -q/two + sqrt(DD)
        temp2 = -q/two - sqrt(DD)
        u = abs(temp1)**(1.D0/3.D0)
        v = abs(temp2)**(1.D0/3.D0)
        if(temp1 .lt. 0.D0) u=-u
        if(temp2 .lt. 0.D0) v=-v
        y1  = u + v
        y2r = -(u+v)/two
        y2i =  (u-v)*sqrt(3.D0)/two
      endif
! Step 4: Final transformation -----------------------------------------
      temp1 = b/a/3.D0
      y1 = y1-temp1
      y2 = y2-temp1
      y3 = y3-temp1
      y2r=y2r-temp1
! Assign answers -------------------------------------------------------
      if(DD .lt. 0.D0)then
        call sortm(y1,y2,y3,y1,y2,y3)
        r1 = dcmplx( y1,  0.D0)
        r2 = dcmplx( y2,  0.D0)
        r3 = dcmplx( y3,  0.D0) 
        del=3
      elseif(DD .eq. 0.D0)then
        call sortm(y1,y2r,y2r,y1,y2r,y2r)
        r1 = dcmplx( y1,  0.D0)
        r2 = dcmplx(y2r,  0.D0)
        r3 = dcmplx(y2r,  0.D0)
        del=3
      else
        !IF(abs(y2i).le.1.D-7)y2i=0.D0  
        r1 = dcmplx( y1,  0.D0)
        r2 = dcmplx(y2r, y2i)
        r3 = dcmplx(y2r,-y2i)
        del=1
      endif
      return
      end subroutine root3


!*********************************************************************************************
      subroutine root4(b,c,d,e,r1,r2,r3,r4,reals)
!********************************************************************************************* 
!* PURPOSE:      This subroutine aim on solving quartic equations: x^4+b*x^3+c*x^2+d*x+e=0.
!* INPUTS:       b, c, d, e-----they are the coefficients of equation. 
!* OUTPUTS:      r1,r2,r3,r4----roots of the equation, with complex number form.
!*               reals------------the number of real roots among r1,r2,r3,r4.  
!* ROUTINES CALLED:  root3   
!* AUTHOR:         Yang, Xiao-lin & Wang, Jian-cheng (2012)
!* DATE WRITTEN:   1 Jan 2012 
!*********************************************************************************************
      implicit none
      Double precision b,c,d,e,q,r,s,realp,imagep,two
      parameter(two=2.D0)
      complex*16 r1,r2,r3,r4,s1,s2,s3,temp(1:4),temp1
      integer i,j,del,reals
   
      reals=0
      q=c-3.D0*b**2/8.D0
      r=d-b*c/two+b**3/8.D0
      s=e-b*d/4.D0+b**2*c/16.D0-3.D0*b**4/256.D0
      call root3(two*q,q**2-4.D0*s,-r**2,s1,s2,s3,del) 
      !write(*,*)'roots3=',s1,s2,s3,del

      If(del.eq.3)then
          If(real(s3).ge.0.D0)then
              reals=4
              s1=dcmplx(real(sqrt(s1)),0.D0)  
              s2=dcmplx(real(sqrt(s2)),0.D0)
              s3=dcmplx(real(sqrt(s3)),0.D0)
          else
              reals=0
              s1=sqrt(s1)  
              s2=sqrt(s2) 
              s3=sqrt(s3)
          endif
      else
          If(real(s1).ge.0.D0)then
              reals=2
              s1=dcmplx(real(sqrt(s1)),0.D0)
              s2=sqrt(s2)
              s3=dcmplx(real(s2),-aimag(s2))   
          else
              reals=0
              s1=sqrt(s1)  
              s2=sqrt(s2) 
              s3=sqrt(s3)
          endif 
      endif 
      if(real(s1*s2*s3)*(-r) .lt. 0.D0)then
          s1=-s1
      end if
      temp(1)=(s1+s2+s3)/two-b/4.D0
      temp(2)=(s1-s2-s3)/two-b/4.D0
      temp(3)=(s2-s1-s3)/two-b/4.D0
      temp(4)=(s3-s2-s1)/two-b/4.D0

      Do i=1,4
          Do j=1+i,4
              If(real(temp(i)).gt.real(temp(j)))then
                  temp1=temp(i)
                  temp(i)=temp(j)
                  temp(j)=temp1
              endif  
          enddo  
      enddo
      r1=temp(1)
      r2=temp(2)
      r3=temp(3)
      r4=temp(4)
      !write(*,*)'roots4=',temp
      return
      end subroutine root4


!*******************************************************************************
      subroutine sortm(a1,a2,a3,s1,s2,s3)
!*******************************************************************************
!* PURPOSE:  This subroutine aim on sorting a1, a2, a3 by a decreasing way.
!* INPUTS:    a1,a2,a3----they are the number list required to bo sorted. 
!* OUTPUTS:   s1,s2,s3----sorted number list with decreasing way. 
!*      
!* AUTHOR:     Yang, Xiao-lin & Wang, Jian-cheng (2012)
!* DATE WRITTEN:  1 Jan 2012 
!******************************************************************************* 
      implicit none
  
      Double precision s1,s2,s3,s4,temp,arr(1:3),a1,a2,a3
      integer i,j
   
      arr(1)=a1
      arr(2)=a2
      arr(3)=a3
   
      Do i=1,3
          Do j=i+1,3
              If(arr(i)<arr(j))then
                 temp=arr(i)
                 arr(i)=arr(j)
                 arr(j)=temp
              end if     
          end do 
      end do
      s1=arr(1)
      s2=arr(2)
      s3=arr(3)
      end subroutine sortm
!**************************************
      end module rootfind 
!************************************** 


!*********************************************************************************************** 
      module ellfunction
!***********************************************************************************************
!*    PURPOSE:  This module includes supporting functions and subroutines to compute 
!*              Weierstrass' and Jacobi's elliptical integrals and functions by Carlson's 
!*              integral method. Those codes mainly come from Press (2007) and geokerr.f of
!*              Dexter & Agol (2009).   
!*    AUTHOR:     Yang, Xiao-lin & Wang, Jian-cheng (2012)
!*    DATE WRITTEN:  1 Jan 2012 
!***********************************************************************************************
      use constants
      use rootfind
      implicit none

      contains
!********************************************************************************************** 
      Double precision function weierstrassP(z,g2,g3,r1,del)
!********************************************************************************************** 
!*     PURPOSE:  Compute weierstrassP-elliptic functions \WP(z,g2,g3). One must
!*             note that input parameter z and the return value is real number.    
!*     ARGUMENTS: 
!*     INPUTS:    z -- the independent variable.
!*                g2, g3 -- are paramters of U(t)=4*t^3-g2*t-g3. U(t) is the standard form
!*                               of polynomial of Weierstrass's elliptic integeral.
!*                r1(3) -- complex numbers, are roots of equation 4*t^3-g2*t-g3=0.
!*                del -- the number of real roots of equation 4*t^3-g2*t-g3=0.                                                  
!*     REMARKS:  
!*     AUTHOR:  Xiaolin, Yang.
!*     DATE WRITTEN:  6 Nov 2012.
!*     REVISIONS:
!*********************************************************************************************
      implicit none
      Double precision  z,g2,g3,g1,e1,e2,e3,k2,u,sn,alp,bet,sig,lamb,cn,dn,realp,&
                        imagep,rfx,rfy,rfz,EK,two,four,zero,halfperiodwp
      parameter(two=2.D0,four=4.D0,zero=0.D0)  
      complex*16 r1(1:3) 
      integer  i,del
 
      If(z.eq.zero)then
          weierstrassP=infinity
      else          
          z=abs(z)  
          if(del.eq.3)then
              e1=real(r1(1))
              e2=real(r1(2))
              e3=real(r1(3))
              k2=(e2-e3)/(e1-e3) 
              u=z*sqrt(e1-e3)
              call sncndn(u,1.D0-k2,sn,cn,dn)
              weierstrassP=e3+(e1-e3)/sn**2
          else
              alp=-real(r1(1))/two
              bet=abs(aimag(r1(2)))
              sig=(9.D0*alp**2+bet**2)**(one/four)
              lamb=(sig**2-three*alp)/bet
              k2=0.5D0+1.5D0*alp/sig**2 
              u=two*sig*z
              call sncndn(u,1.D0-k2,sn,cn,dn)    
              if(cn.gt.zero)then  
                  weierstrassP = two*sig**two*(1.D0+sqrt(1.D0-sn**2))/sn**2-two*alp-sig**2
              else
                  IF(abs(sn).gt.1.D-7)THEN
                      weierstrassP = two*sig**two*(1.D0-sqrt(1.D0-sn**2))/sn**2-two*alp-sig**2 
                  ELSE 
                      weierstrassP = -two*alp
                  ENDIF
              endif 
          end if
      endif 
      return
      end function weierstrassP


!*************************************************************************************************
      Function halfperiodwp(g2,g3,rts,del)
!************************************************************************************************* 
!*    PURPOSE:   to compute the real semi period of Weierstrass' elliptical function 
!*               \wp(z;g_2,g_3) and all of this function involved are real numbers.   
!*    INPUTS:    g_2, g_3---two parameters. 
!*               rts(1:3)----an array which are the roots of equation W(t)=4t^3-g_2t-g_3=0.
!*               del--------number of real roots among rts(1:3). 
!*    RETURN:    halfperiodwp----the semi period of function \wp(z;g_2,g_3).  
!*    ROUTINES CALLED:  rf
!*    AUTHOR:    Yang, Xiao-lin & Wang, Jian-cheng (2012)
!*    DATE WRITTEN:  1 Jan 2012 
!***************************************************************************************
      implicit none
      Double precision halfperiodwp,g2,g3,g1,e1,e2,e3,zero,one,two,&
             three,four,EK,alp,bet,sig,lamb,k2
      parameter(zero=0.D0,one=1.D0,two=2.D0,three=3.D0,four=4.D0) 
      complex*16 r1,r2,r3,rts(1:3)  
      integer  i,del
   
      if(del.eq.3)then
          e1=real(rts(1))
          e2=real(rts(2))
          e3=real(rts(3))
          k2=(e2-e3)/(e1-e3)
          EK=rf(zero,one-k2,one)
          halfperiodwp=EK/sqrt(e1-e3)
      else
          alp=-real(rts(1))/two
          bet=abs(aimag(r2))
          sig=(9*alp**two+bet**two)**(one/four)
          k2=one/two+three/two*alp/sig**two
          EK=rf(zero,one-k2,one)
          halfperiodwp=EK/sig
      endif
      return
      End Function halfperiodwp


!************************************************************************ 
      subroutine sncndn(uu,emmc,sn,cn,dn) 
!************************************************************************
!*     PURPOSE:  Compute Jacobi-elliptic functions SN,CN,DN.
!*     ARGUMENTS:  Given the arguments U,EMMC=1-k^2 calculate 
!*                      sn(u,k), cn(u,k), dn(u,k).
!*     REMARKS:  
!*     AUTHOR:  Press et al (1992).
!*     DATE WRITTEN:  25 Mar 91.
!*     REVISIONS:
!************************************************************************
      implicit none      
      Double precision uu,emmc,sn,CA
      Double precision ,optional :: cn,dn
      parameter (CA=3.D0-8)
      integer i,ii,l
      Double precisiona,b,c,d,emc,u,em(13),en(13)
      logical bo

      emc=emmc
      u=uu
      if(emc.ne.0.D0)then
          bo=(emc.lt.0.D0) 
          if(bo)then
              d=1.D0-emc   !t'=t*k, u'=k*u, k'^2=1./k^2,  
              emc=-emc/d
              d=sqrt(d)
              u=d*u
          end if 
          a=1.D0
          dn=1.D0
          l1: do i=1,13 
              l=i
              em(i)=a
              emc=sqrt(emc)
              en(i)=emc
              c=0.5D0*(a+emc)
              if(abs(a-emc).le.CA*a) exit
              emc=a*emc
              a=c
          end do l1
          u=c*u
          sn=sin(u)  
          cn=cos(u)  
          if(sn.eq.0.D0)then
              if(bo)then
                  a=dn
                  dn=cn
                  cn=a
                  sn=sn/d
              end if
              return 
          endif 
          a=cn/sn
          c=a*c
          l2: do ii=l,1,-1
              b=em(ii)
              a=c*a
              c=dn*c
              dn=(en(ii)+a)/(b+a)
              a=c/b
          enddo l2
          a=1.D0/sqrt(c**2+1.D0)
          if(sn.lt.0.D0)then
              sn=-a
          else
              sn=a
          endif
              cn=c*sn
          return    
      else
          cn=1.D0/cosh(u)
          dn=cn                   
          sn=tanh(u)
          return
      end if
      end subroutine sncndn


!*********************************************************************************
      Double precision FUNCTION rf(x,y,z) 
!********************************************************************************* 
!*     PURPOSE: Compute Carlson fundamental integral RF
!*              R_F=1/2 \int_0^\infty dt (t+x)^(-1/2) (t+y)^(-1/2) (t+z)^(-1/2)
!*     ARGUMENTS: Symmetric arguments x,y,z
!*     ROUTINES CALLED:  None.
!*     ALGORITHM: Due to B.C. Carlson.
!*     ACCURACY:  The parameter ERRTOL sets the desired accuracy.
!*     REMARKS:  
!*     AUTHOR:  Press et al (2007).
!*     DATE WRITTEN:  25 Mar 91.
!*     REVISIONS:
!********************************************************************************* 
      implicit none
      Double precision x,y,z,ERRTOL,TINY1,BIG,THIRD,C1,C2,C3,C4,delta,zero
      parameter (ERRTOL=0.0025D0,TINY1=1.5D-38,BIG=3.D37,THIRD=1.D0/3.D0,C1=1.D0/24.D0,&
                 C2=0.1D0,C3=3.D0/44.D0,C4=1.D0/14.D0,zero=0.D0)
      Double precision alamb,ave,delx,dely,delz,e2,e3,sqrtx,sqrty,sqrtz,xt,yt,zt
 
      IF(x.lt.zero)x=zero
      IF(y.lt.zero)y=zero
      IF(z.lt.zero)z=zero
      xt=x
      yt=y
      zt=z
      sqrtx=sqrt(xt)
      sqrty=sqrt(yt)
      sqrtz=sqrt(zt)
      alamb=sqrtx*(sqrty+sqrtz)+sqrty*sqrtz
      xt=0.25D0*(xt+alamb)
      yt=0.25D0*(yt+alamb)
      zt=0.25D0*(zt+alamb)
      ave=THIRD*(xt+yt+zt)
      delx=(ave-xt)/ave
      dely=(ave-yt)/ave
      delz=(ave-zt)/ave 
      delta=max(abs(delx),abs(dely),abs(delz))
      Do while(delta.gt.ERRTOL)
          sqrtx=sqrt(xt)
          sqrty=sqrt(yt)
          sqrtz=sqrt(zt)
          alamb=sqrtx*(sqrty+sqrtz)+sqrty*sqrtz
          xt=0.25D0*(xt+alamb)
          yt=0.25D0*(yt+alamb)
          zt=0.25D0*(zt+alamb)
          ave=THIRD*(xt+yt+zt)
          delx=(ave-xt)/ave
          dely=(ave-yt)/ave
          delz=(ave-zt)/ave
          delta=max(abs(delx),abs(dely),abs(delz))
      enddo
      e2=delx*dely-delz**2
      e3=delx*dely*delz
      rf=(1.D0+(C1*e2-C2-C3*e3)*e2+C4*e3)/sqrt(ave)
      return 
      end Function rf


!*************************************************************************** 
      Double precision FUNCTION rj(x,y,z,p) 
!*************************************************************************** 
!*     PURPOSE: Compute Carlson fundamental integral RJ
!*     RJ(x,y,z,p) = 3/2 \int_0^\infty dt
!*                      (t+x)^(-1/2) (t+y)^(-1/2) (t+z)^(-1/2) (t+p)^(-1)
!*     ARGUMENTS: x,y,z,p
!*     ROUTINES CALLED:  RF, RC.
!*     ALGORITHM: Due to B.C. Carlson.
!*     ACCURACY:  The parameter ERRTOL sets the desired accuracy.
!*     REMARKS:  
!*     AUTHOR:  Press et al (1992)
!*     DATE WRITTEN:  25 Mar 91.
!*     REVISIONS:
!************************************************************************** 
      implicit none
      Double precision x,y,z,p,ERRTOL,TINY1,BIG,C1,C2,C3,C4,C5,C6,C7,C8,delta,zero
      Double precision a,alamb,alpha,ave,b,beta,delp,delx,dely,delz,ea,eb,ec,ed,ee,fac,pt,&
             rcx,rho,sqrtx,sqrty,sqrtz,sum1,tau,xt,yt,zt
      Parameter(ERRTOL=0.0015D0,TINY1=2.5D-13,BIG=9.D11,C1=3.D0/14.D0,&
                C2=1.D0/3.D0,C3=3.D0/22.D0,C4=3.D0/26.D0,C5=0.75D0*C3,&
                C6=1.5D0*C4,C7=0.5D0*C2,C8=C3+C3,zero=0.D0)

      IF(x.lt.zero)x=zero
      IF(y.lt.zero)y=zero
      IF(z.lt.zero)z=zero
      sum1=0.D0
      fac=1.D0
      If(p.gt.0.D0)then
          xt=x
          yt=y
          zt=z
          pt=p
      else
          xt=min(x,y,z)
          zt=max(x,y,z)
          yt=x+y+z-xt-zt 
          a=1.D0/(yt-p)
          b=a*(zt-yt)*(yt-xt)
          pt=yt+b
          rho=xt*zt/yt
          tau=p*pt/yt
          rcx=rc(rho,tau)
      endif
          sqrtx=sqrt(xt)
          sqrty=sqrt(yt)
          sqrtz=sqrt(zt)
          alamb=sqrtx*(sqrty+sqrtz)+sqrty*sqrtz
          alpha=(pt*(sqrtx+sqrty+sqrtz)+sqrtx*sqrty*sqrtz)**2
          beta=pt*(pt+alamb)**2
          sum1=sum1+fac*rc(alpha,beta)
          fac=0.25D0*fac
          xt=0.25D0*(xt+alamb)
          yt=0.25D0*(yt+alamb)
          zt=0.25D0*(zt+alamb)
          pt=0.25D0*(pt+alamb)
          ave=0.2D0*(xt+yt+zt+pt+pt)
          delx=(ave-xt)/ave
          dely=(ave-yt)/ave
          delz=(ave-zt)/ave
          delp=(ave-pt)/ave
          delta=max(abs(delx),abs(dely),abs(delz),abs(delp))
      Do while(delta.gt.ERRTOL)
          sqrtx=sqrt(xt)
          sqrty=sqrt(yt)
          sqrtz=sqrt(zt)
          alamb=sqrtx*(sqrty+sqrtz)+sqrty*sqrtz
          alpha=(pt*(sqrtx+sqrty+sqrtz)+sqrtx*sqrty*sqrtz)**2
          beta=pt*(pt+alamb)**2
          sum1=sum1+fac*rc(alpha,beta)
          fac=0.25D0*fac
          xt=0.25D0*(xt+alamb)
          yt=0.25D0*(yt+alamb)
          zt=0.25D0*(zt+alamb)
          pt=0.25D0*(pt+alamb)
          ave=0.2D0*(xt+yt+zt+pt+pt)
          delx=(ave-xt)/ave
          dely=(ave-yt)/ave
          delz=(ave-zt)/ave
          delp=(ave-pt)/ave
          delta=max(abs(delx),abs(dely),abs(delz),abs(delp))
      enddo  
      ea=delx*(dely+delz)+dely*delz
      eb=delx*dely*delz
      ec=delp**2
      ed=ea-3.D0*ec
      ee=eb+2.D0*delp*(ea-ec)
      rj=3.D0*sum1+fac*(1.D0+ed*(-C1+C5*ed-C6*ee)+eb*(C7+&
                   delp*(-C8+delp*C4))+delp*ea*(C2-delp*C3)-&
      C2*delp*ec)/(ave*sqrt(ave)) 
      If(p.le.0.D0)rj=a*(b*rj+3.D0*(rcx-rf(xt,yt,zt)))
      return
      end Function rj


!************************************************************************ 
      FUNCTION rc(x,y)
!************************************************************************
!*     PURPOSE: Compute Carlson degenerate integral RC
!*              R_C(x,y)=1/2 \int_0^\infty dt (t+x)^(-1/2) (t+y)^(-1)
!*     ARGUMENTS: x,y
!*     ROUTINES CALLED:  None.
!*     ALGORITHM: Due to B.C. Carlson.
!*     ACCURACY:  The parameter ERRTOL sets the desired accuracy.
!*     REMARKS:  
!*     AUTHOR:  Press et al (1992)
!*     DATE WRITTEN:  25 Mar 91.
!*     REVISIONS:
!************************************************************************
      implicit none
      Double precision rc,x,y,ERRTOL,TINY1,sqrtNY,BIG,TNBG,COMP1,COMP2,THIRD,C1,C2,C3,C4
      PARAMETER(ERRTOL=0.0012D0,TINY1=1.69D-38,sqrtNY=1.3D-19,BIG=3.D37,&
      TNBG=TINY1*BIG,COMP1=2.236D0/sqrtNY,COMP2=TNBG*TNBG/25.D0,&
      THIRD=1.D0/3.D0,C1=0.3D0,C2=1.D0/7.D0,C3=0.375D0,C4=9.D0/22.D0)
      Double precisionalamb,ave,s,w,xt,yt
 
      if(y.gt.0.D0)then
          xt=x
          yt=y
          w=1.D0
      else
          xt=x-y
          yt=-y
          w=sqrt(x)/sqrt(xt)
      endif
          alamb=2.D0*sqrt(xt)*sqrt(yt)+yt
          xt=0.25D0*(xt+alamb)
          yt=0.25D0*(yt+alamb)
          ave=THIRD*(xt+yt+yt)
          s=(yt-ave)/ave
      Do While(abs(s).gt.ERRTOL)
          alamb=2.D0*sqrt(xt)*sqrt(yt)+yt
          xt=0.25D0*(xt+alamb)
          yt=0.25D0*(yt+alamb)
          ave=THIRD*(xt+yt+yt)
          s=(yt-ave)/ave
      ENDdo
      rc=w*(1.D0+s*s*(C1+s*(C2+s*(C3+s*C4))))/sqrt(ave)
      return
      END FUNCTION rc


!**************************************************************************************** 
      FUNCTION rd(x,y,z)
!****************************************************************************************  
!*     PURPOSE: Compute Carlson degenerate integral RD
!*              R_D(x,y,z)=3/2 \int_0^\infty dt (t+x)^(-1/2) (t+y)^(-1/2) (t+z)^(-3/2)
!*     ARGUMENTS: x,y,z
!*     ROUTINES CALLED:  None.
!*     ALGORITHM: Due to B.C. Carlson.
!*     ACCURACY:  The parameter ERRTOL sets the desired accuracy.
!*     REMARKS:  
!*     AUTHOR:  Press et al (1992)
!*     DATE WRITTEN:  25 Mar 91.
!*     REVISIONS:
!***************************************************************************************
      implicit none
      Double precision rd,x,y,z,ERRTOL,TINY,BIG,C1,C2,C3,C4,C5,C6,zero
      PARAMETER (ERRTOL=0.0015D0,TINY=1.D-25,BIG=4.5D21,C1=3.D0/14.D0,C2=1.D0/6.D0,&
                 C3=9.D0/22.D0,C4=3.D0/26.D0,C5=0.25D0*C3,C6=1.5D0*C4,zero=0.D0)
      !Computes Carlson's elliptic integral of the second kind, RD(x; y; z). x and y must be
      !nonnegative, and at most one can be zero. z must be positive. TINY must be at least twice
      !the negative 2/3 power of the machine overflow limit. BIG must be at most 0:1 ERRTOL
      !time_0 the negative 2/3 power of the machine underflow limit.
      Double precision alamb,ave,delx,dely,delz,ea,eb,ec,ed,ee,fac,sqrtx,sqrty,sqrtz,sum,xt,yt,zt
      !if(min(x,y).lt.0.D0.or.min(x+y,z).lt.TINY.or. max(x,y,z).gt.BIG)then
      !     rd=0.D0 
      !     return
      !endif
      IF(x.lt.zero)x=zero
      IF(y.lt.zero)y=zero 
          xt=x
          yt=y
          zt=z
          sum=0.D0
          fac=1.D0
 
          sqrtx=sqrt(xt)
          sqrty=sqrt(yt)
          sqrtz=sqrt(zt)
          alamb=sqrtx*(sqrty+sqrtz)+sqrty*sqrtz
          sum=sum+fac/(sqrtz*(zt+alamb))
          fac=0.25D0*fac
          xt=0.25D0*(xt+alamb)
          yt=0.25D0*(yt+alamb)
          zt=0.25D0*(zt+alamb)
          ave=0.2D0*(xt+yt+3.D0*zt)
          delx=(ave-xt)/ave
          dely=(ave-yt)/ave
          delz=(ave-zt)/ave
      DO While(max(abs(delx),abs(dely),abs(delz)).gt.ERRTOL)
          sqrtx=sqrt(xt)
          sqrty=sqrt(yt)
          sqrtz=sqrt(zt)
          alamb=sqrtx*(sqrty+sqrtz)+sqrty*sqrtz
          sum=sum+fac/(sqrtz*(zt+alamb))
          fac=0.25D0*fac
          xt=0.25D0*(xt+alamb)
          yt=0.25D0*(yt+alamb)
          zt=0.25D0*(zt+alamb)
          ave=0.2D0*(xt+yt+3.D0*zt)
          delx=(ave-xt)/ave
          dely=(ave-yt)/ave
          delz=(ave-zt)/ave
      End DO
          ea=delx*dely
          eb=delz*delz
          ec=ea-eb
          ed=ea-6.D0*eb
          ee=ed+ec+ec
          rd=3.D0*sum+fac*(1.D0+ed*(-C1+C5*ed-C6*delz*ee)+delz*(C2*ee+&
                           delz*(-C3*ec+delz*C4*ea)))/(ave*sqrt(ave))
          return
      END function rd 


!******************************************************************** 
      Function EllipticF(t,k2)
!******************************************************************** 
!*     PURPOSE: calculate Legendre's first kind elliptic integral: 
!*              F(t,k2)=\int_0^t dt/sqrt{(1-t^2)*(1-k2*t^2)}.  
!*     ARGUMENTS: t, k2
!*     ROUTINES CALLED:  RF  
!*     AUTHOR:  Press et al (1992)
!*     DATE WRITTEN:  25 Mar 91.
!*     REVISIONS:
!********************************************************************  
      implicit none
      Double precision t,k2,EllipticF,x1,y1,z1
      !Perpous: calculate Legendre's first kind elliptic integrals: 
      !F(t,k2)=\int_0^t dt/sqrt{(1-t^2)*(1-k2*t^2)}. 
 
      x1=1.D0-t*t
      y1=1.D0-k2*t*t
      z1=1.D0
      !Press et al. 2007 (6.12.19)
      EllipticF=t*rf(x1,y1,z1)
      return
      end function EllipticF  


!********************************************************************* 
      Function EllipticE(t,k2)
!********************************************************************* 
!*     PURPOSE: calculate Legendre's second kind elliptic integrals: 
!*              E(t,k2)=\int_0^t sqrt{1-k2*t^2}/sqrt{(1-t^2)}dt.
!*     ARGUMENTS: t, k2
!*     ROUTINES CALLED:  RF, RD 
!*     AUTHOR:  Press et al (1992)
!*     DATE WRITTEN:  25 Mar 91.
!*     REVISIONS:
!********************************************************************** 
      implicit none
      Double precision t,k2,EllipticE,x1,y1,z1
      !Perpous: calculate Legendre's second kind elliptic 
      !integrals: E(t,k2)=\int_0^t sqrt{1-k2*t^2}/sqrt{(1-t^2)}dt.
 
      x1=1.D0-t*T
      y1=1.D0-k2*t*t
      z1=1.D0
      !Press et al. 2007 (6.12.20)
      EllipticE=t*rf(x1,y1,z1)-1.D0/3.D0*k2*t**3*rd(x1,y1,z1)
      return
      end function EllipticE


!*********************************************************************** 
      Function EllipticPI(t,n,k2)
!*********************************************************************** 
!*     PURPOSE: calculate Legendre's third kind elliptic integrals: 
!*              PI(t,n,k2)=\int_0^t /(1+nt^2)/sqrt{(1-k2*t^2)(1-t^2)}dt. 
!*     ARGUMENTS: t, k2
!*     ROUTINES CALLED:  RF, RJ
!*     AUTHOR:  Press et al (1992)
!*     DATE WRITTEN:  25 Mar 91.
!*     REVISIONS:
!*********************************************************************** 
      implicit none
      Double precision t,k2,EllipticPI,x1,y1,z1,w1,n
      !Perpous: calculate Legendre's third kind elliptic integrals: 
      !PI(t,n,k2)=\int_0^t /(1+nt^2)/sqrt{(1-k2*t^2)(1-t^2)}dt.
 
      x1=1.D0-t*t
      y1=1.D0-k2*t*t
      z1=1.D0
      w1=1.D0+n*t*t
      !Press et al. 2007 (6.12.20)
      EllipticPI=t*rf(x1,y1,z1)-1.D0/3.D0*n*t*t*t*rj(x1,y1,z1,w1)
      return
      end function EllipticPI


!**************************************************************************************************** 
      subroutine weierstrass_int_J3(y,x,bb,del,a4,b4,p5,rff_p,integ,cases)
!****************************************************************************************************
!*     PURPOSE: Computes integrals: J_k(h)=\int^x_y (b4*t+a4)^(k/2)*(4*t^3-g_2*t-g_3)^(-1/2)dt.
!*              Where integer index k can be 0, -2, -4 and 2. (112) and (113) of Yang & Wang (2013).   
!*     INPUTS:  x,y -- limits of integral.
!*              bb(1:3) -- Roots of equation 4*t^3-g_2*t-g_3=0 solved by routine root3.
!*              del -- Number of real roots in bb(1:3).
!*              p4,rff_p,integ -- p4(1:4) is an array which specifies the value of index k of J_k(h). 
!*                 If p5(1)=0, then J_0 was computed and sent to integ(1).
!*                 If p5(1)=-1, then J_0 was replaced by parameter rff_p, i.e., J_0 = rff_p.  
!*                 If p5(2)=-2, then J_{-2} was computed and sent to integ(2).                      
!*                 If p5(3)=2, then J_{2} was computed and sent to integ(3).
!*                 If p5(4)=-4, then J_{-4} was computed and sent to integ(4).
!*                 If p5(5)=4, then J_{4} was computed and sent to integ(5).
!*              cases -- If cases=1, then only J_0 was computed.
!*                       If cases=2, then only J_0 and J_{-2} are computed.
!*                       If cases=3, then only J_0, J_{-2} and J_{2} are computed.    
!*                       If cases=4, then J_0, J_{-2}, J_{2} and J_{-4} are computed.            
!*     OUTPUTS: integ -- is an array saved the results of J_k(h), and a4 = -h. 
!*     ROUTINES CALLED:  ellcubicreals, ellcubiccomplexs     
!*     ACCURACY:   Machine.
!*     REMARKS: Based on Yang & Wang (2012).
!*     AUTHOR:     Yang & Wang (2012).
!*     DATE WRITTEN:  4 Jan 2012 
!****************************************************************************************************
      implicit none
      Double precision y,x,yt,xt,h4,g2,g3,a1,b1,a2,b2,a3,b3,a4,b4,rff_p,integ(5),b,three,two,one,&
                       tempt,f,g,h,a44,b44,sign_h,integ4(4)
      parameter  (three=3.0D0,two=2.0D0,one=1.D0)
      integer  del,p5(5),i,cases,p4(4)
      complex*16 bb(1:3)
      logical :: inverse,neg

      xt=x
      yt=y
      a44=a4
      b44=b4
      inverse=.false.
      neg=.false.
      If(abs(xt-yt).eq.0.D0)then
          integ=0.D0
          return 
      endif
      if(yt.gt.xt)then
          tempt=xt
          xt=yt
          yt=tempt 
          inverse=.true.   
      endif
      b=0.D0
      sign_h=sign(one,b44*xt+a44)
      If(del.eq.3)then
          a44=sign_h*a44
          b44=sign_h*b44
          a1=-real(bb(1))
          a2=-real(bb(2))
          a3=-real(bb(3))
          b1=1.D0
          b2=1.D0
          b3=1.D0
          call ellcubicreals(p5,a1,b1,a2,b2,a3,b3,a44,b44,yt,xt,rff_p*two,integ,cases)
          if(inverse)then
              integ(1)=-integ(1)
              integ(2)=-integ(2)
              integ(3)=-integ(3)
              integ(4)=-integ(4)
              integ(5)=-integ(5)
          endif
          Do i=1,5
              integ(i)=integ(i)/two 
              integ(i)=integ(i)*(sign_h)**(-p5(i)/2)
          Enddo
      else
          a44=sign_h*a44
          b44=sign_h*b44
          a1=-real(bb(1))
          b1=one
          f=real(bb(2))**2+aimag(bb(2))**2
          g=-two*real(bb(2))
          h=one
          IF( yt.lt.real(bb(1)) )THEN
              yt = real(bb(1))
          ENDIF  
          call ellcubiccomplexs(p5,a1,b1,a44,b44,f,g,h,yt,xt,rff_p*two,integ,cases)
          if(inverse)then
              integ(1)=-integ(1)
              integ(2)=-integ(2)
              integ(3)=-integ(3)
              integ(4)=-integ(4)
              integ(5)=-integ(5)
          endif
          Do i=1,5
              integ(i)=integ(i)/two 
              integ(i)=integ(i)*(sign_h)**(-p5(i)/2)
          Enddo
      endif 
      return
      end subroutine weierstrass_int_J3


!******************************************************************************************************************
      subroutine carlson_doublecomplex5m(y,x,f1,g1,h1,f2,g2,h2,a5,b5,p5,rff_p,integ,cases)
!******************************************************************************************************************    
!*     PURPOSE: Computes integrals: J_k(h)=\int^x_y (b5*r+a5)^(k/2)*[(h1*r^2+g1*r+f1)(h2*r^2+g2*r+f2)]^(-1/2)dr.
!*              Where integer index k can be 0, -2, -4, 2 and 4. (114) of Yang & Wang (2012).   
!*     INPUTS:  x,y -- limits of integral.  
!*              p5,rff_p,integ -- p5(1:5) is an array which specifies the value of index k of J_k(h). 
!*                 If p5(1)=0, then J_0 was computed and sent to integ(1).
!*                 If p5(1)=-1, then J_0 was replaced by parameter p, and rff_p=p.  
!*                 If p5(2)=-2, then J_{-2} was computed and sent to integ(2).                      
!*                 If p5(3)=2, then J_{2} was computed and sent to integ(3).
!*                 If p5(4)=-4, then J_{-4} was computed and sent to integ(4).
!*                 If p5(4)=4, then J_{4} was computed and sent to integ(5).
!*              cases -- If cases=1, then only J_0 will be computed.
!*                       If cases=2, then only J_0 and J_{-2} will be computed.
!*                       If cases=3, then only J_0, J_{-2} and J_{2} will be computed.
!*                       If cases=4, then J_0, J_{-2}, J_{2} and J_{-4} will be computed. 
!*                       If cases=5, then J_0, J_{-2}, J_{2} and J_{4} will be computed.     
!*     OUTPUTS: integ -- is an array saved the results of J_k(h). 
!*     ROUTINES CALLED:  elldoublecomplexs
!*     ACCURACY:   Machine.
!*     REMARKS: Based on Yang & Wang (2012).
!*     AUTHOR:     Yang & Wang (2012).
!*     DATE WRITTEN:  4 Jan 2012 
!*********************************************************************************************************************
      implicit none
      Double precision y,x,xt,yt,f1,g1,h1,f2,g2,h2,a5,b5,rff,integ(1:5),b,tempt,f,g,h,a55,&
             b55,sign_h,one,zero,rff_p
      parameter(one=1.D0,zero=0.D0)
      integer  reals,p5(1:5),cases,i
      logical :: inverse,neg
      xt=x
      yt=y
      a55=a5
      b55=b5
      inverse=.false.
      neg=.false.
      If(abs(xt-yt).eq.0.D0)then
          integ=zero
          return 
      endif
      if(yt.gt.xt)then
          tempt=xt
          xt=yt
          yt=tempt 
          inverse=.true.   
      endif
      sign_h=sign(one,b55*xt+a55) 
      a55=sign_h*a55
      b55=sign_h*b55
      call elldoublecomplexs(p5,f1,g1,h1,f2,g2,h2,a55,b55,yt,xt,rff_p,integ,cases)
      if(inverse)then
          integ(1)=-integ(1)
          integ(2)=-integ(2)
          integ(3)=-integ(3)
          integ(4)=-integ(4)
          integ(5)=-integ(5)
      endif 
      Do i=1,5 
          integ(i)=integ(i)*(sign_h)**(-p5(i)/2)
      Enddo
      return
      end subroutine carlson_doublecomplex5m


!*****************************************************************************************************
      subroutine ellcubicreals(index_p5,a1,b1,a2,b2,a3,b3,a4,b4,y,x,rff_p,integ,cases)
!*****************************************************************************************************
!*     PURPOSE: Computes J_k(h)=\int_y^x dt (b4*t+a4)^(k/2)[(b1*t+a1)*(b2*t+a2)*(b3*t+a3)]^{-1/2}. 
!*              It is the case of equation W(t)=4*t^3-g_2*t-g_3=0 has three real roots. 
!*              Equation (112) of Yang & Wang (2013).   
!*     INPUTS:  Arguments for above integral. If index_p4(1)=0, then J_0 will be computed, 
!*              else J_0 will be replaced by parameter rff_p, i.e., J_0 = rff_p. 
!*     OUTPUTS:  Value of integral J_k(h), which are saved in an array integ(1:5).
!*     ROUTINES CALLED: RF,RJ,RC,RD
!*     ACCURACY:   Machine.
!*     AUTHOR:     Dexter & Agol (2009)
!*     MODIFIED:   Yang & Wang (2012)
!*     DATE WRITTEN:  4 Mar 2009
!*     REVISIONS: 
!*****************************************************************************************************
      implicit NONE
      Double precision zero,one,half,two,three,ellcubic,d12,d13,d14,d24,d34,X1,X2,X3,X4,&
             Y1,Y2,Y3,Y4,U1c,U32,U22,W22,U12,Q22,P22,I1c,I3c,r12,r13,r24i,r34i,&
             I2c,J2c,K2c,a1,b1,a2,b2,a3,b3,a4,b4,y,x,rff_p,r14,r24,r34,rff,&
             integ(5),A111,J1c,rdd
      integer  index_p5(5),cases
      PARAMETER ( ZERO=0.D0, ONE=1.D0, TWO=2.D0, HALF=0.5d0, THREE=3.D0 )
      ellcubic=0.d0
 !c (2.1) Carlson (1989)
      d12=a1*b2-a2*b1
      d13=a1*b3-a3*b1
      d14=a1*b4-a4*b1
      d24=a2*b4-a4*b2
      d34=a3*b4-a4*b3
      r14=a1/b1-a4/b4
      r24=a2/b2-a4/b4
      r34=a3/b3-a4/b4  
 !c (2.2) Carlson (1989)
      X1=dsqrt(abs(a1+b1*x))
      X2=dsqrt(abs(a2+b2*x))
      X3=dsqrt(abs(a3+b3*x))
      X4=dsqrt(abs(a4+b4*x))
      Y1=dsqrt(abs(a1+b1*y))
      Y2=dsqrt(abs(a2+b2*y))
      Y3=dsqrt(abs(a3+b3*y))
      Y4=dsqrt(abs(a4+b4*y))
 !c! (2.3) Carlson (1989)
      If(x.lt.infinity)then 
          U1c=(X1*Y2*Y3+Y1*X2*X3)/(x-y)
          U12=U1c**2
          U22=((X2*Y1*Y3+Y2*X1*X3)/(x-y))**2
          U32=((X3*Y1*Y2+Y3*X1*X2)/(x-y))**2 
      else
          U1c=dsqrt(abs(b2*b3))*Y1
          U12=U1c**2
          U22=b1*b3*Y2**2
          U32=b2*b1*Y3**2 
      endif   
 !c (2.4) Carlson (1989)
      W22=U12
      W22=U12-b4*d12*d13/d14
 ! (2.5) Carlson (1989)
      If(x.lt.infinity)then  
          Q22=(X4*Y4/X1/Y1)**2*W22 
      else
          Q22=b4/b1*(Y4/Y1)**2*W22  
      endif 
      P22=Q22+b4*d24*d34/d14
       
 !c Now, compute the three integrals we need [-1,-1,-1],[-1,-1,-1,-2], and 
 !c  [-1,-1,-1,-4]:we need to calculate the [-1,-1,-1,2] integral,we add it in this part.
      if(index_p5(1).eq.0) then
 !c (2.21) Carlson (1989)
        rff=rf(U32,U22,U12)
        integ(1)=two*rff
        if(cases.eq.1)return
      else
          rff=rff_p/two
      endif
        !c (2.12) Carlson (1989)
        I1c=two*rff
        IF(index_p5(3).eq.2 .or. index_p5(5).eq.4 .or. index_p5(4).eq.-4)THEN
            rdd = rd(U32,U22,U12)
            !c (2.13) Carlson (1989)
            I2c = two/three*d12*d13*rdd+two*X1*Y1/U1c
        ENDIF 
        If(index_p5(3).eq.2) then 
             !  (2.39) Carlson (1989)
                integ(3)=(b4*I2c-d14*I1c)/b1 
                if(cases.eq.3)return
        endif

        If(index_p5(5).eq.4) then 
            !  (2.6) Carlson (1989) 
            A111=X1*X2*X3-Y1*Y2*Y3 
            !  (2.16) Carlson (1989) 
            J1c=d12*d13*I1c-two*b1*A111      
            !  (2.46) Carlson (1989)
            integ(5)=(b4*b4/three/b1)*(-two*(r14+r24+r34)*I2c+three*b1*r14*r14*I1c&
                           -J1c/b1/b2/b3)
            if(cases.eq.5)return
        endif

        If(X1*Y1.ne.zero) then
        !c (2.14) Carlson (1989)
          I3c=two*rc(P22,Q22)-two*b1*d12*d13/three/d14*rj(U32,U22,U12,W22)
        Else
        ! One can read the paragraph between (2.19) and (2.20) of Carlson (1989).
          I3c=-two*b1*d12*d13/three/d14*rj(U32,U22,U12,W22)
        Endif
        if(index_p5(2).eq.-2) then
        !c (2.49) Carlson (1989)
            integ(2)=(b4*I3c-b1*I1c)/d14
            if(Y4*X4.eq.zero)integ(2) = infinity*two
            if(cases.eq.2)return
        endif

          If(index_p5(4).eq.-4)then
              if(Y4*X4.eq.zero)then
                  integ(4) = two*infinity
                  return  
              endif
                !c (2.1)  Carlson (1989)
                r12=a1/b1-a2/b2
                r13=a1/b1-a3/b3
                r24i=b2*b4/(a2*b4-a4*b2)
                r34i=b3*b4/(a3*b4-a4*b3) 
             If(x.lt.infinity)then
                !c (2.17) Carlson (1989)
                J2c=two/three*d12*d13*rdd+two*d13*X2*Y2/X3/Y3/U1c
                !c (2.59) & (2.6) Carlson (1989)
                K2c=b3*J2c-two*d34*(X1*X2/X3/X4**2-Y1*Y2/Y3/Y4**2)
             else
                J2c=two/three*d12*d13*rdd+two*d13*Y2/b3/Y3/Y1
                K2c=b3*J2c+two*d34*Y1*Y2/Y3/Y4**2
             endif  
             !c (2.62) Carlson (1989)
             integ(4)=-I3c*half/d14*(one/r14+one/r24+one/r34)+&
                  half*b4/d14/d24/d34*K2c+(b1/d14)**2*(one-half*r12*r13*r24i*r34i)*I1c
         endif  
      return
      end  subroutine ellcubicreals


!************************************************************************************************
  subroutine ellcubiccomplexs(index_p5,a1,b1,a4,b4,f,g,h,y,x,rff_p,integ,cases)
!************************************************************************************************
!*     PURPOSE: Computes J_k(h)=\int_y^x dt (b4*t+a4)^(k/2)[(b1*t+a1)*(h*t^2+g*t+f)]^{-1/2}. 
!*              It is the case of equation W(t)=4*t^3-g_2*t-g_3=0 has one real root and one pair of. 
!*              complex root. Equation (114) of Yang & Wang (2013).  
!*     INPUTS:  Arguments for above integral.
!*     OUTPUTS:  Value of integral. If index_p4(1)=0, then J_0 will be computed, 
!*               else J_0 will be replaced by parameter p, and rff_p=p. 
!*     ROUTINES CALLED: RF,RJ,RC,RD
!*     ACCURACY:   Machine.
!*     AUTHOR:     Dexter & Agol (2009)
!*     MODIFIED:   Yang & Wang (2012)
!*     DATE WRITTEN:  4 Mar 2009
!*     REVISIONS: 
!************************************************************************************************   
      Double precision a1,b1,a4,b4,f,g,h,y,x,X1,X4,Y1,Y4,d14,d34,beta1,beta4,a11,c44,integ(5),&
             a142,xi,eta,M2,Lp2,Lm2,I1c,U,U2,Wp2,W2,Q2,P2,rho,I3c,I2c,r24xr34,r12xr13,&
             N2c,K2c,ellcubic,zero,one,two,four,three,half,six,rff,rdd,r14,Ay1,rff_p,&
             A111,J1c            
      integer   index_p5(5),cases
      PARAMETER ( ONE=1.D0, TWO=2.D0, HALF=0.5d0, THREE=3.D0, FOUR=4.d0, SIX=6.d0, ZERO=0.D0 )
      ellcubic=0.d0
      X1=dsqrt(abs(a1+b1*x))
      X4=dsqrt(abs(a4+b4*x)) 
      Y1=dsqrt(abs(a1+b1*y)) 
      Y4=dsqrt(abs(a4+b4*y)) 
      r14=a1/b1-a4/b4 
      d14=a1*b4-a4*b1
 !c (2.2) Carlson (1991)
      beta1=g*b1-two*h*a1
      beta4=g*b4-two*h*a4
 !c (2.3) Carlson (1991)
      a11=sqrt(two*f*b1*b1-two*g*a1*b1+two*h*a1*a1)
      c44=sqrt(two*f*b4*b4-two*g*a4*b4+two*h*a4*a4)
      a142=two*f*b1*b4-g*(a1*b4+a4*b1)+two*h*a1*a4
 !c (2.4) Carlson (1991)
      xi=sqrt(f+g*x+h*x*x)
      eta=sqrt(abs(f+g*y+h*y*y))
 !c (3.1) Carlson (1991):
      if(x.lt.infinity)then
          M2=((X1+Y1)*sqrt((xi+eta)**two-h*(x-y)**two)/(x-y))**two
      else 
          M2=b1*(two*sqrt(h)*eta+g+two*h*y) 
      endif  
 !c (3.2) Carlson (1991):
      Lp2=M2-beta1+sqrt(two*h)*a11
      Lm2=M2-beta1-sqrt(two*h)*a11

      if(index_p5(1).eq.0) then
 !c (1.2)   Carlson (1991)
        rff=rf(M2,Lm2,Lp2)
        integ(1)=four*rff
        if(cases.eq.1)return
      else
        rff=rff_p/four  
      endif
  !c (3.8)  1991
         I1c=rff*four
  !c (3.3) 1991
         if(x.lt.infinity)then
             U=(X1*eta+Y1*xi)/(x-y)
         else
             U=sqrt(h)*Y1
         endif 
          !c (3.5) 1991
          rho=sqrt(two*h)*a11-beta1
          IF(index_p5(3).eq.2 .or. index_p5(5).eq.4 .or. index_p5(4).eq.-4)THEN
              rdd=rd(M2,Lm2,Lp2)
              !  (3.9) Carlson (1991)
              I2c=a11*sqrt(two/h)/three*(four*rho*rdd-six*rff+three/U)+two*X1*Y1/U 
          ENDIF
          If(index_p5(3).eq.2)  then
              !  (2.39) Carlson (1989)
              integ(3)=(b4*I2c-d14*I1c)/b1 
              if(cases.eq.3)return 
          endif 
          If(index_p5(5).eq.4) then 
              !  (2.5) Carlson (1991) 
              A111=X1*xi-Y1*eta
              !  (3.12) Carlson (1991)  
              J1c=a11*a11*I1c/two-two*b1*A111     
              !  (2.46) Carlson (1989)
              !  (2.19) 1991 b2*b3=h, 1/r24+1/r34=two*b4*beta4/c44**two,
              !  r24+r34=beta4/h/b4 
              integ(5)=(b4*b4/three/b1)*(-two*(r14+beta4/h/b4)*I2c+three*b1*r14*r14*I1c&
                           -J1c/b1/h)
              if(cases.eq.5)return
          endif


         U2=U*U
         Wp2=M2-b1*(a142+a11*c44)/d14
         W2=U2-a11**two*b4/two/d14
      !c (3.4) 1991
         if(x.lt.infinity)then   
            Q2=(X4*Y4/X1/Y1)**two*W2
          else
            Q2=(b4/b1)*(Y4/Y1)**two*W2
         endif 
        P2=Q2+c44**two*b4/two/d14
        !c!! (3.9) 1991
        If(X1*Y1.ne.0.D0) then
          I3c=(two*a11/three/c44)*((-four*b1/d14)*(a142+a11*c44)*rj(M2,Lm2,Lp2,Wp2)&
                                -six*rff+three*rc(U2,W2))+two*rc(P2,Q2)
        Else
          I3c=(two*a11/three/c44)*((-four*b1/d14)*(a142+a11*c44)*rj(M2,Lm2,Lp2,Wp2)&
                                -six*rff+three*rc(U2,W2))
        Endif

        if(index_p5(2).eq.-2) then
          !c (2.49) Carlson (1989)  
          integ(2)=(b4*I3c-b1*I1c)/d14
          if(Y4*X4.eq.zero)integ(2) = two*infinity
          if(cases.eq.2)return
        endif

        If(index_p5(4).eq.-4)  then
              if(Y4*X4.eq.zero)then
                  integ(4) = two*infinity
                  return  
              endif
             ! (2.19) Carlson (1991)
             r24Xr34=half*c44**two/h/b4**two
             r12Xr13=half*a11**two/h/b1**two
             !c (3.11) Carlson (1991) 
             N2c=two/three*sqrt(two*h)/a11*(four*rho*rdd-six*rff+three/U)!+two/X1/Y1/U
             If(Y1.eq.zero)then
                Ay1=-two*b1*xi/X1
                If(x.ge.infinity)Ay1=zero
             else
                Ay1=(two*d14*eta/Y4**two+a11**two/X1/U)/Y1
             endif  
             !c (2.5) & (3.12) Carlson (1991) 
             K2c=half*a11**two*N2c-two*d14*(xi/X1/X4**two)+Ay1
             !c (2.62) Carlson (1989)
             integ(4)=-I3c*half/d14*(one/r14+two*b4*beta4/c44**two)+&
                  half/d14/(h*b4*r24Xr34)*K2c+(b1/d14)**two*(one-half*r12Xr13/r24Xr34)*I1c 
         endif 
      return
      end  subroutine ellcubiccomplexs


!********************************************************************************************
   subroutine  elldoublecomplexs(index_p5,f1,g1,h1,f2,g2,h2,a5,b5,y,x,rff_p,integ,cases)
!********************************************************************************************
!*     PURPOSE: Computes J_k(h)=\int_y^x dt (f_1+g_1t+h_1t^2)^{p_1/2} 
!*                       (f_2+g_2t+h_2t^2)^{p_2/2} (a_5+b_5t)^{p_5/2}. 
!*     INPUTS:  Arguments for above integral.
!*     OUTPUTS:  Value of integral. If index_p5(1)=0, then J_0 will be computed, 
!*               else J_0 will be replaced by parameter p, and p = rff_p. 
!*     ROUTINES CALLED: RF,RJ,RC,RD
!*     ACCURACY:   Machine.    
!*     AUTHOR:     Dexter & Agol (2009)
!*     MODIFIED:   Yang & Wang (2012)
!*     DATE WRITTEN:  4 Mar 2009
!*     REVISIONS: [-1,-1,-1,-1,0]=integ(1),[-1,-1,-1,-1,-2]=integ(2),
!*                [-1,-1,-1,-1,-4]=integ(4),[-1,-1,-1,-1,2]=integ(3),
!*                [-1,-1,-1,-1,4]=integ(5)
!*******************************************************************************************
      implicit NONE 
      Double precision one,half,two,three,four,six,f1,g1,h1,f2,g2,h2,a5,b5,y,x,xi1,xi2,eta1,eta2,integ(5),&
                       theta1,theta2,zeta1,zeta2,M,M2,delta122,delta112,delta222,delta,deltap,lp2,lm2,&
                       deltam,rff,ellquartic,U,U2,alpha15,beta15,alpha25,beta25,lambda,omega2,psi,xi5,eta5,&
                       gamma1,gamma2,Am111m1,A1111m4,XX,S,mu,T,V2,b2,a2,H,A1111m2,xi1p,B,G,Sigma,Lambda0,&
                       Omega02,psi0,X0,mu0,b02,a02,H0,S2,T2,eta1p,T02,V02,psi2,T0,A1111,rff_p
      integer  index_p5(5),cases
      one=1.D0
      half=0.5D0
      two=2.D0
      three=3.D0
      four=4.D0
      six=6.D0
      !c (2.1) Carlson (1992)
      If(x.lt.infinity)then
          xi1=sqrt(f1+g1*x+h1*x**two)
          xi2=sqrt(f2+g2*x+h2*x**two)
      else
          xi1=x*sqrt(h1)
          xi2=x*sqrt(h2) 
      endif 
      eta1=sqrt(abs(f1+g1*y+h1*y*y))
      eta2=sqrt(abs(f2+g2*y+h2*y*y))
       !c (2.4) Carlson (1992)
      If(x.lt.infinity)then
         theta1=two*f1+g1*(x+y)+two*h1*x*y
         theta2=two*f2+g2*(x+y)+two*h2*x*y
      else
         theta1=(g1+two*h1*y)*x
         theta2=(g2+two*h2*y)*x
      endif 
        !c (2.5) Carlson (1992)
      zeta1=sqrt(two*xi1*eta1+theta1)
      zeta2=sqrt(two*xi2*eta2+theta2)
        !c (2.6) Carlson (1992)
      If(x.lt.infinity)then
          M=zeta1*zeta2/(x-y)
          M2=M*M
      else
          M2=(two*sqrt(h1)*eta1+g1+two*h1*y)*(two*sqrt(h2)*eta2+g2+two*h2*y)
      endif

         !c (2.7) Carlson (1992)
      delta122=two*f1*h2+two*f2*h1-g1*g2
      delta112=four*f1*h1-g1*g1
      delta222=four*f2*h2-g2*g2
      Delta=sqrt(delta122*delta122-delta112*delta222)
        !c (2.8) Carlson (1992)
      Deltap=delta122+Delta
      Deltam=delta122-Delta
      Lp2=M2+Deltap
      Lm2=M2+Deltam
 
      if(index_p5(1).eq.0) then
        rff=rf(M2,Lm2,Lp2)
        !c (2.36) Carlson (1992)
        integ(1)=four*rff
        if(cases.eq.1)return
      else
        rff=rff_p/four 
      endif
        !c (2.6) Carlson (1992)
        If(x.lt.infinity)then
          U=(xi1*eta2+eta1*xi2)/(x-y)
          U2=U*U
        else
          U=sqrt(h1)*eta2+sqrt(h2)*eta1
          U2=U*U 
        endif
        
        !c (2.11) Carlson (1992)
        alpha15=two*f1*b5-g1*a5 
        alpha25=two*f2*b5-g2*a5
        beta15=g1*b5-two*h1*a5 
        beta25=g2*b5-two*h2*a5
        !c (2.12) Carlson (1992)
        gamma1=half*(alpha15*b5-beta15*a5)
        gamma2=half*(alpha25*b5-beta25*a5)
        !c (2.13) Carlson (1992)
        Lambda=delta112*gamma2/gamma1
        Omega2=M2+Lambda
        psi=half*(alpha15*beta25-alpha25*beta15)
        psi2=psi*psi
        !c (2.15) Carlson (1992)
        xi5=a5+b5*x
        eta5=a5+b5*y
        !c (2.16) Carlson (1992)
        If(x.lt.infinity)then
                Am111m1=one/xi1*xi2-one/eta1*eta2
                A1111m4=xi1*xi2/xi5**two-eta1*eta2/eta5**two
                A1111m2=xi1*xi2/xi5-eta1*eta2/eta5 
                XX = (xi5*eta5*theta1*half*Am111m1-xi1*xi2*eta5**two+&
                                  eta1*eta2*xi5**two)/(x-y)**two
                mu=gamma1*xi5*eta5/xi1/eta1
        else
                Am111m1=sqrt(h2/h1)-one/eta1*eta2
                A1111m4=sqrt(h1*h2)/b5**two-eta1*eta2/eta5**two 
                A1111m2=sqrt(h1*h2)*x/b5-eta1*eta2/eta5
                XX=b5*eta5*h1*y*Am111m1-eta5**two*sqrt(h1*h2)+eta1*eta2*b5**two
                mu=gamma1*b5*eta5/sqrt(h1)/eta1
        endif

        !c (2.17) Carlson (1992)
    
        !c (2.18) Carlson (1992)
        S=half*(M2+delta122)-U2
        S2=S*S
        !c (2.19) Carlson (1992)
        T=mu*S+two*gamma1*gamma2
        T2=T*T
        V2=mu**two*(S2+Lambda*U2)
        !c (2.20) Carlson (1992)
        b2=Omega2**two*(S2/U2+Lambda)
        a2=b2+Lambda**two*psi2/gamma1/gamma2
        !c (2.22) Carlson (1992)
        H=delta112*psi*(rj(M2,Lm2,Lp2,Omega2)/three+half*rc(a2,b2))/gamma1**two-XX*rc(T2,V2)
        If(index_p5(3).eq.2 .or. index_p5(5).eq.4)then
                !(2.23)--(2.29) Carlson (1992)
                psi=g1*h2-g2*h1
                Lambda=delta112*h2/h1
                Omega2=M2+Lambda
                Am111m1=one/xi1*xi2-one/eta1*eta2
                A1111=xi1*xi2-eta1*eta2
                XX=(theta1*half*Am111m1-A1111)/(x-y)**two
                b2=Omega2**two*(S2/U2+Lambda)
                a2=b2+Lambda**two*psi**two/h1/h2
                mu=h1/xi1/eta1
                T=mu*S+two*h1*h2
                T2=T*T
                V2=mu**two*(S2+Lambda*U2)
                H0=delta112*psi*(rj(M2,Lm2,Lp2,Omega2)/three+&
                          half*rc(a2,b2))/h1**two-XX*rc(T2,V2)
            If(index_p5(3).eq.2)then
                !(2.42) Carlson (1992)
                integ(3)=two*b5*H0-two*beta15*rff/h1
                if(cases.eq.3)return
            endif
            If(index_p5(5).eq.4)then 
                !c (2.2) Carlson (1992)
                If(x.lt.infinity)then
                  xi1p=half*(g1+two*h1*x)/xi1
                else
                  xi1p=sqrt(h1)
                endif
                  eta1p=half*(g1+two*h1*y)/eta1
                  !c (2.3) Carlson (1992)
                  B=xi1p*xi2-eta1p*eta2
                  !c (2.9) Carlson (1992)
                If(x.lt.infinity)then
                  G=two/three*Delta*Deltap*rd(M2,Lm2,Lp2)+half*Delta/U+&
                       (delta122*theta1-delta112*theta2)/four/xi1/eta1/U  
                else
                  G=two/three*Delta*Deltap*rd(M2,Lm2,Lp2)+half*Delta/U+&
                      (delta122*(g1+two*h1*y)-delta112*(g2+two*h2*y))/four/sqrt(h1)/eta1/U  
                endif
                !c (2.10) Carlson (1992)  
                Sigma=G-Deltap*rff+B
                !(2.44) Carlson (1992)
                integ(5)=-b5*(beta15/h1+beta25/h2)*H0+b5**two*&
                            Sigma/h1/h2+beta15**two*rff/h1**two
                if(cases.eq.5)return
            endif 
        endif
            if (index_p5(2).eq.-2) then
                !c (2.39) Carlson (1992)
                integ(2)=-two*(b5*H+beta15*rff/gamma1)
                if(xi5*eta5.eq.zero)integ(2) = infinity 
                if(cases.eq.2)return
            endif
            If(index_p5(4).eq.-4)then
                if(xi5*eta5.eq.zero)then
                    integ(4) = infinity 
                    return
                endif
                !c (2.2) Carlson (1992)
                If(x.lt.infinity)then
                  xi1p=half*(g1+two*h1*x)/xi1
                else
                  xi1p=sqrt(h1)
                endif
                  eta1p=half*(g1+two*h1*y)/eta1
                  !c (2.3) Carlson (1992)
                  B=xi1p*xi2-eta1p*eta2
                  !c (2.9) Carlson (1992)
                If(x.lt.infinity)then
                  G=two/three*Delta*Deltap*rd(M2,Lm2,Lp2)+half*Delta/U+&
                         (delta122*theta1-delta112*theta2)/four/xi1/eta1/U  
                else
                  G=two/three*Delta*Deltap*rd(M2,Lm2,Lp2)+half*Delta/U+&
                     (delta122*(g1+two*h1*y)-delta112*(g2+two*h2*y))/four/sqrt(h1)/eta1/U  
                endif
                !c (2.10) Carlson (1992)  
                Sigma=G-Deltap*rff+B
                !c (2.41) Carlson (1992)
                integ(4)=b5*(beta15/gamma1+beta25/gamma2)*H+beta15**two*rff/gamma1**two+&
                         b5**two*(Sigma-b5*A1111m2)/gamma1/gamma2
            endif
      integ=ellquartic
      return
      end  subroutine  elldoublecomplexs


!********************************************************************************************
      end module ellfunction



!********************************************************************************************
      module blcoordinates
!*******************************************************************************
!*     PURPOSE: This module aims on computing 4 Boyer-Lindquist coordinates (r,\theta,\phi,t)
!*              and affine parameter \sigam as functions of p, i.e., r(p), \mu(p), \phi(p),
!*              t(p) and \sigma(p).    
!*     ACCURACY:   Machine.    
!*     AUTHOR:     Yang & Wang (2012)  
!*     DATE WRITTEN:  4 Jan 2012 
!***********************************************************************

      use constants
      use rootfind
      use ellfunction 
      implicit none

      contains
!************************************************************************************************************** 
      SUBROUTINE YNOGKM(p,kvec,lambda,q,mve,ep,sin_ini,cos_ini,a_spin,e,r_ini,&
                                    radi,mu,time,phi,sigma,cir_orbt,theta_star) 
!**************************************************************************************************************
!*     PURPOSE:  Computes four Boyer-Lindquist coordinates (r,\mu,\phi,t) and affine parameter 
!*               \sigma as functions of parameter p, i.e. functions r(p), \mu(p), \phi(p), t(p)
!*               and \sigma(p). Cf. discussions in Yang & Wang (2012).    
!*     INPUTS:   p---------------------independent variable, which must be nonnegative.
!*               kvec(4)---------------an array contains k_{r}, k_{\theta}, k_{\phi}, and k_{t}, which are 
!*                                     defined by equations (92)-(96). k_{i} can also be regarded as 
!*                                     components of four-momentum of a photon measured under the LNRF frame. 
!*                                     This array can be computed by subroutine lambdaq(...), see below.     
!*               lambda,q,mve,ep-------constants of motion, defined by lambda=L_z/E, q=Q/E^2,
!*                                     mve = \mu_m/E, ep=epsilon/varepsilon. 
!*               sin_ini,cos_ini-------sin_ini=sin(\theta_{ini}), cos_ini=cos(\theta_{ini}), where 
!*                                     \theta_{ini} is the initial \theta coordinate of the particle.
!*               a_spin,e--------------the spin and the electric charge of the black hole, 
!*                                     and -1 =< a_spin^2+e^2 <= 1 must be satisfied. 
!*               r_ini-----------------the initial radial coordinate of the particle. 
!*               cir_orbt--------------If cir_orbt = .True. to compute the spherical orbits.
!*                                     If cir_orbt = .False. to compute the non-spherical orbits.
!*               theta_star------------If cir_orbt = .True., theta_star should be specified, which is the
!*                                     theta coordinate of the turning points.
!************************************************************************************************************       
!*     OUTPUTS:  radi-----------value of function r(p). 
!*               mu-------------value of function \mu(p).
!*               time-----------value of function t(p). 
!*               phi------------value of function \phi(p).
!*               sigma----------value of function \sigma(p).
!*               tm1,tm2--------number of time_0 of particle meets turning points \mu_tp1 and \mu_tp2
!*                              respectively for a given p. 
!*               tr1,tr2--------number of time_0 of particle meets turning points r_tp1 and r_tp2
!*                              respectively for a given p.            
!*     ROUTINES CALLED: INTRPART, INTTPART.
!*     ACCURACY:   Machine.    
!*     AUTHOR:     Yang & Wang (2012)  
!*     DATE WRITTEN:  5 Jan 2012
!*     REVISIONS: ****************************************** 
      IMPLICIT NONE 
      DOUBLE PRECISION kvec(4),lambda,q,sin_ini,cos_ini,a_spin,r_ini,radi,mu,time,phi,sigma,&
             zero,one,two,three,four,phi_r,time_r,aff_r,phi_t,time_t,p,mu_tp,mu_tp2,Rab,&
             rp,mup,p_int,mu_cos,r_coord,mve,aff_t,ep,e,theta_star
      CHARACTER varble
      PARAMETER(zero=0.D0, one=1.D0, two=2.D0, three=3.D0, four=4.D0)
      LOGICAL rotate,mobseqmtp,cir_orbt
      INTEGER tm1,tm2,tr1,tr2,reals,del,t1,t2
  
      IF(.not.cir_orbt)THEN
! call integrat_r_part to evaluate t_r,\phi_r,\sigma_r, and function r(p) (here is r_coord). 
          call INTRPART(p,kvec(1),kvec(2),lambda,q,mve,ep,a_spin,e,r_ini,cos_ini,phi_r,time_r,aff_r,r_coord,tr1,tr2) 
! call integrat_theta_part to evaluate t_\mu,\phi_\mu,\sigma_\mu, and function \mu(p) (here is mu_cos).
          call INTTPART(p,kvec(3),kvec(2),lambda,q,mve,sin_ini,cos_ini,a_spin,phi_t,time_t,mu_cos,tm1,tm2)  

          radi=r_coord
          aff_t=time_t  
      ELSE 
          cos_ini = zero !Set cos_ini = 0, sin_ini = 1, means that we always assume the initial position  
          sin_ini = one  !of the particle is in the equatorial plane.
          call CIR_ORBIT_PTA(p,kvec(3),kvec(2),lambda,q,mve,ep,sin_ini,cos_ini,a_spin,e,&
                        r_ini,theta_star,phi_t,time_t,aff_t,mu_cos,t1,t2)   
          radi=r_ini
          time_r=zero
          aff_r=zero
          phi_r=zero
      ENDIF  
 
      mu=mu_cos
      !time coordinate value **************************************************************
      time=time_r+time_t
      !affine parameter value *************************************************************
      sigma=aff_r+aff_t
      !phi coordinate value ***************************************************************
      rotate=.false. 
      !write(*,*)'phi2=',kvec(3),lambda!phi_r,phi_t,kvec(3),tm1,tm2!time_r,time_t!tm1,tm2!,tm1,tm2,lambda,kvec(3)
      IF(ABS(cos_ini).NE.ONE)THEN
          phi=(phi_r+phi_t)
          !write(*,*)'phi',phi,phi_r,phi_t,cos_ini,tm1,tm2
          IF(lambda.EQ.zero)THEN
              !If(mu_tp.eq.one.and.mu_tp2.eq.-one)phi=phi+(t1+t2)*PI
              phi=phi+(tm1+tm2)*PI
          ENDIF 
          phi=DMOD(phi,twopi)
          IF(phi.LT.zero)THEN
              phi=phi+twopi
          ENDIF
      ELSE 
          phi=-(phi_t+phi_r+(tm1+tm2)*PI)
          Rab=dsqrt(kvec(3)**two+kvec(2)**two)
          IF(phi.NE.zero)THEN
              rotate=.TRUE.
          ENDIF
          IF(Rab.NE.zero)THEN
              if((kvec(2).ge.zero).and.(kvec(3).gt.zero))then
          ! a cos_ini was multiplied to control the rotate direction
                  phi=cos_ini*phi+asin(kvec(3)/Rab)  
              endif
              if((kvec(2).lt.zero).and.(kvec(3).ge.zero))then
                  phi=cos_ini*phi+PI-asin(kvec(3)/Rab)
              endif
              if((kvec(2).le.zero).and.(kvec(3).lt.zero))then
                  phi=cos_ini*phi+PI-asin(kvec(3)/Rab)
              endif
              if((kvec(2).gt.zero).and.(kvec(3).le.zero))then
                  phi=cos_ini*phi+twopi+asin(kvec(3)/Rab)
              endif
          ELSE
              phi=zero
          ENDIF
          IF(rotate)THEN
              phi=Mod(phi,twopi)
              IF(phi.LT.zero)THEN
                  phi=phi+twopi
              ENDIF
          ENDIF
      ENDIF      
      RETURN
      END SUBROUTINE YNOGKM


!************************************************************************
      Function r_ms(theta,a_spin,e)
!*******************************************************************************
!*     PURPOSE: ---------Computes the radius of inner most stable spherical orbit (ISSO)
!*                       r_ms for the given theta_star, a_spin and e. 
!*     INPUTS: ----------Arguments for above purpose.
!*     OUTPUTS:----------r_ms. 
!*     ROUTINES CALLED:  
!*     ACCURACY:   Machine.    
!*     AUTHOR:     Yang & Wang (2012)
!*     DATE WRITTEN:  4 Mar 2013 
!***********************************************************************
      implicit none
      Double precision theta,M,a_spin,e,r1,r2,rc,f1,f2,fc,Delta,r_ms,zero
  
      If(e*e+a_spin*a_spin.gt.1.D0)THEN
          Write(*,*)'r_ms():The valuse of a^2+e^2 you input is >1, which is not valid, the code '
          write(*,*)'should be stopped. Please input a valid one and try it again.'
          STOP
      ENDIF
      zero = 0.D0
      r1 = 13.D0
      f1 = d_temp_F_rms(r1,theta,a_spin,e)

      Do while(f1.gt.0.D0)
          r1 = r1-1.D-3
          f1 = d_temp_F_rms(r1,theta,a_spin,e)
      Enddo
      r2 = r1+1.D-3
      f2 = d_temp_F_rms(r2,theta,a_spin,e)

 
          Do while(dabs(r1-r2).lt.1.D-10)
              rc = (r1+r2)*0.5D0
              fc = d_temp_F_rms(rc,theta,a_spin,e) 
              If(fc*f1 .gt. zero)then
                  r1 = rc
                  f1 = fc
              else
                  r2 = rc 
                  f2 = fc
              endif 
          Enddo 
          r_ms = (r1+r2)*0.5D0
      Return
      End Function r_ms


!************************************************************************
      Function r_ph(theta,a_spin,e)
!*******************************************************************************
!*     PURPOSE: ---------Computes the radius of photon spherical orbit  
!*                       r_ph for the given theta_star, a_spin and electric charge e. 
!*     INPUTS: ----------Arguments for above purpose.
!*     OUTPUTS:----------r_ph
!*     ROUTINES CALLED:  
!*     ACCURACY:   Machine.    
!*     AUTHOR:     Yang & Wang (2012)
!*     DATE WRITTEN:  4 Mar 2013 
!***********************************************************************
      implicit none
      Double precision theta,M,a_spin,e,r1,r2,Delta,r_ph
  
      r1 = 40.D0
      r2 = temp_F_rph(r1,theta,a_spin,e)
      Delta = r2-r1
      r1 = r2
      Do While(dabs(Delta).gt.1.D-5)
          r2 = temp_F_rph(r1,theta,a_spin,e)
          Delta = r2-r1
          r1 = r2
      Enddo
      r_ph = r1
      Return
      End FUNCTION r_ph
 

!************************************************************************
      Function r_mb(theta,a_spin,e)
!*******************************************************************************
!*     PURPOSE: ---------Computes the radius of Marginally bound spherical orbit  
!*                       r_mb for the given theta_star, a_spin and electric charge e. 
!*     INPUTS: ----------Arguments for above purpose.
!*     OUTPUTS:----------r_mb
!*     ROUTINES CALLED:  
!*     ACCURACY:   Machine.    
!*     AUTHOR:     Yang & Wang (2012)
!*     DATE WRITTEN:  4 Mar 2013 
!***********************************************************************
      implicit none
      Double precision theta,M,a_spin,e,r1,r2,Delta,r_mb
  
      r1 = 40.D0
      r2 = temp_F_mb(r1,theta,a_spin,e)
      Delta = r2-r1
      r1 = r2
      Do While(dabs(Delta).gt.1.D-5)
          r2 = temp_F_mb(r1,theta,a_spin,e) 
          Delta = r2-r1
          r1 = r2
      Enddo
      r_mb = r1
      Return
      End FUNCTION r_mb


!************************************************************************
      Function r_mse(theta_star,a_spin)
!*******************************************************************************
!*     PURPOSE: ---------Computes the radius of inner most stable spherical orbit (ISSO)
!*                       r_mse for the given theta_star, a_spin. e = 0. 
!*     INPUTS: ----------Arguments for above purpose.
!*     OUTPUTS:----------r_mse. 
!*     ROUTINES CALLED:  
!*     ACCURACY:   Machine.    
!*     AUTHOR:     Yang & Wang (2012)
!*     DATE WRITTEN:  4 Mar 2013 
!***********************************************************************
      implicit none
      Double precision theta_star,a_spin,e,r1,r2,Delta,r_mse
  
      r1 = 40.D0
      r2 = temp_rms_F(r1,theta_star,a_spin)
      Delta = r2-r1
      r1 = r2
      Do While(dabs(Delta).gt.1.D-20)
          r2 = temp_rms_F(r1,theta_star,a_spin) 
          Delta = r2-r1
          r1 = r2
      Enddo
      r_mse = r1
      Return
      End Function


!*******************************************************************************
      Function temp_rms_F(r,theta,a_spin)
!*******************************************************************************
!*     PURPOSE: Computes a temp function F(...) in order to calculate r_ms
!*     INPUTS:  Arguments for above purpose.
!*     OUTPUTS:  Value of F(...) 
!*     ACCURACY:   Machine.    
!*     AUTHOR:     Yang & Wang (2012)
!*     MODIFIED:  
!*     DATE WRITTEN:  4 Mar 2013
!***********************************************************************
      use constants
      Implicit none
      Double precision r,theta,M,a_spin,e,temp_rms_F,a2,cs,si,e2,&
             Sigma,P,Delta,Evsm2 

      e = zero
      M = one
      a2 = a_spin*a_spin
      e2 = e*e
      cs = dcos(theta*dtor)
      si = dsin(theta*dtor)
      Delta = r*r-two*M*r+a2+e2
      Sigma = r*r+a2*cs*cs
      P = M*(r*r-a2*cs*cs)-r*e2
      Evsm2 = (a_spin*si*dsqrt(P)+(Delta-a2*si*si)*dsqrt(r))**two/Sigma/&
                         (-P+r*(Delta-a2*si*si)+two*a_spin*si*dsqrt(r*P))
      temp_rms_F = ( ( a_spin**4.D0*cs**4.D0+6.D0*r*r*a2*cs*cs+two*M*r*&
                              (r*r-3.D0*a2*cs*cs)/(1.D0-Evsm2)&
                     )/3.D0&
                   )**(1.D0/4.D0) 
      Return
      End Function

!**********************************************************************
      Function temp_F_rph(r,theta,a_spin,e)
!*******************************************************************************
!*     PURPOSE: Computes a temp function F(...) in order to calculate r_photon
!*     INPUTS:  Arguments for above purpose.
!*     OUTPUTS:  Value of F(...) 
!*     ACCURACY:   Machine.    
!*     AUTHOR:     Yang & Wang (2012)
!*     MODIFIED:  
!*     DATE WRITTEN:  4 Mar 2013
!***********************************************************************
      use constants
      Implicit none
      Double precision r,theta,M,a_spin,e,temp_F_rph,a2,cs,si,e2 

      M = one
      a2 = a_spin*a_spin
      e2 = e*e
      cs = dcos(theta*dtor)
      si = dsin(theta*dtor)
      temp_F_rph = ( -( -M*(r*r-a2*cs*cs)+r*e*e+(-two*M*r+a2+e2-a2*si*si)*r+&
                   two*a_spin*si*dsqrt(r*( M*(r*r-a2*cs*cs)-r*e*e ))&
                  )&
               )**(1.D0/3.D0)
      Return
      End Function


!**********************************************************************
      Function temp_F_mb(r,theta,a_spin,e)
!*******************************************************************************
!*     PURPOSE: Computes a temp function F(...) in order to calculate r_mb
!*     INPUTS:  Arguments for above purpose.
!*     OUTPUTS:  Value of F(...) 
!*     ACCURACY:   Machine.    
!*     AUTHOR:     Yang & Wang (2012)
!*     MODIFIED:  
!*     DATE WRITTEN:  4 Mar 2013
!***********************************************************************
      use constants
      Implicit none
      Double precision r,theta,M,a_spin,e,temp_F_mb,a2,cs,si,e2,&
             Sigma,P,Delta,temp_F1 

      M = one
      a2 = a_spin*a_spin
      e2 = e*e
      cs = dcos(theta*dtor)
      si = dsin(theta*dtor)
      Delta = r*r-two*M*r+a2+e2
      Sigma = r*r+a2*cs*cs
      P = M*(r*r-a2*cs*cs)-r*e2
      temp_F_mb = ( -(-M*r**4.0D0+sigma*(-P+(Delta-a2*si*si)*r)+two*a_spin*si&
                      *dsqrt(r*P)*(Sigma-Delta+a2*si*si)&
                 -a2*si*si*P-(Delta-a2*si*si)**two*r)/M&
               )**(1.D0/4.D0) 
      Return
      End Function temp_F_mb 


!**********************************************************************
      Function temp_F_rms(r,theta,a_spin,e)
!*******************************************************************************
!*     PURPOSE: Computes a temp function F(...) in order to calculate r_ms
!*     INPUTS:  Arguments for above purpose.
!*     OUTPUTS:  Value of F(...) 
!*     ACCURACY:   Machine.    
!*     AUTHOR:     Yang & Wang (2012)
!*     MODIFIED:  
!*     DATE WRITTEN:  4 Mar 2013
!***********************************************************************
      Implicit none
      Double precision r,theta,M,a_spin,e,temp_F_rms,a2,cs,si,dtor,e2,&
             two,Sigma,P,Delta,Evsm2
      Parameter(dtor=2.D0*dasin(1.D0)/180.D0,two=2.D0)

      M = 1.D0
      a2 = a_spin*a_spin
      e2 = e*e
      cs = dcos(theta*dtor)
      si = dsin(theta*dtor)
      Delta = r*r-two*M*r+a2+e2
      Sigma = r*r+a2*cs*cs
      P = M*(r*r-a2*cs*cs)-r*e2
      If(P.lt.0.D0)write(*,*)P
      temp_F_rms = ( a_spin*si*dsqrt(P)+(Delta-a2*si*si)*dsqrt(r) )/dsqrt(Sigma)/&
                         dsqrt(-P+r*(Delta-a2*si*si)+two*a_spin*si*dsqrt(r*P))
 
      Return
      End Function


!**********************************************************************
      Function d_temp_F_rms(r,theta,a_spin,e)
!*******************************************************************************
!*     PURPOSE: Computes a temp function F(...) in order to calculate r_ms
!*     INPUTS:  Arguments for above purpose.
!*     OUTPUTS:  Value of F(...) 
!*     ACCURACY:   Machine.    
!*     AUTHOR:     Yang & Wang (2012)
!*     MODIFIED:  
!*     DATE WRITTEN:  4 Mar 2013
!***********************************************************************
      Implicit none
      Double precision r,theta,M,a_spin,e,d_temp_F_rms,a2,cs,si,dtor,e2,&
             two,Sigma,P,Delta,dr
      Parameter(dtor=2.D0*dasin(1.D0)/180.D0,two=2.D0,dr = 1.D-10)

      M = 1.D0
      a2 = a_spin*a_spin
      e2 = e*e
      cs = dcos(theta*dtor)
      si = dsin(theta*dtor)
      Delta = r*r-two*M*r+a2+e2
      Sigma = r*r+a2*cs*cs
      P = M*(r*r-a2*cs*cs)-r*e2
      d_temp_F_rms =  ( temp_F_rms(r+dr,theta,a_spin,e)-temp_F_rms(r,theta,a_spin,e) )/dr 
      Return
      End Function


!=======================================================================  
      Function rms(a_spin)                      
!***********************************************************************
!*     PURPOSE: Computes inner most stable circular orbit (ISCO) r_{ms}. 
!*     INPUTS:   a_spin ---- Spin of black hole, on interval [-1,1].
!*     OUTPUTS:  radius of inner most stable circular orbit: r_{ms}
!*     ROUTINES CALLED: root4
!*     ACCURACY:   Machine.    
!*     AUTHOR:     Yang & Wang (2012)  
!*     DATE WRITTEN:  4 Jan 2012
!*     REVISIONS: ******************************************************* 
      implicit none
      Double precision rms,a_spin,b,c,d,e
      complex*16 rt(1:4)
      integer  reals,i
      If(a_spin.eq.0.D0)then
          rms=6.D0
          return
      endif
      b=0.D0
      c=-6.D0
      d=8.D0*a_spin
      e=-3.D0*a_spin**2
        ! Bardeen et al. (1972) 
      call root4(b,c,d,e,rt(1),rt(2),rt(3),rt(4),reals)
      Do i=4,1,-1
          If(aimag(rt(i)).eq.0.D0)then
              rms=real(rt(i))**2
              return      
          endif   
      enddo
      end function rms


!****************************************************************************************
      Function rph(a_spin)
!****************************************************************************************
!*     PURPOSE: Computes photon orbit of circluar orbits: r_{ph}. 
!*     INPUTS:   a_spin ---- Spin of black hole, on interval [-1,1].
!*     OUTPUTS:  radius of photon orbit: r_{ph}
!*     ROUTINES CALLED: NONE
!*     ACCURACY:   Machine.    
!*     AUTHOR:     Yang & Wang (2012)  
!*     DATE WRITTEN:  4 Jan 2012
!*     REVISIONS: ****************************************** 
      implicit none
      Double precision rph,a_spin
        ! Bardeen et al. (1972) 

      rph=2.D0*(1.D0+cos(2.D0/3.D0*acos(-a_spin)))
      End function  rph 

!********************************************************************************************
      Function rmb(a_spin)
!********************************************************************************************
!*     PURPOSE: Computes marginally bound orbit of circluar orbits: r_{mb}. 
!*     INPUTS:   a_spin ---- Spin of black hole, on interval [-1,1].
!*     OUTPUTS:  radius of marginally bound orbit: r_{mb}
!*     ROUTINES CALLED: NONE
!*     ACCURACY:   Machine.    
!*     AUTHOR:     Yang & Wang (2012)  
!*     DATE WRITTEN:  4 Jan 2012
!*     REVISIONS: ************************************************************************* 
      implicit none
      Double precision rmb,a_spin

        ! Bardeen et al. (1972)  
      rmb=2.D0-a_spin+2.D0*sqrt(1.D0-a_spin)
      End function  rmb 

!******************************************************************************************************************
      subroutine mutp(kvecp,kvect,sin_ini,cos_ini,a_spin,lambda,q,mve,mu_tp1,mu_tp2,reals,mobseqmtp)
!******************************************************************************************************************
!*     PURPOSE: Returns the coordinates of turning points \mu_tp1 and \mu_tp2 of theta motion, judges
!*                whether the initial theta angle \theta_{ini} is equal to one of turning points, if 
!*                it is true, then mobseqmtp=.TRUE..  
!*     INPUTS:   kvect----------k_{theta}, the initial \theta component of four-momentum of the particle measured 
!*                              under the LNRF, see equation (94) in Yang & Wang (2013), A&A.
!*               sin_ini,cos_ini-------sin_ini=sin(\theta_{ini}), cos_ini=cos(\theta_{ini}), where 
!*                                     \theta_{ini} is the initial \theta coordinate of the particle.
!*               a_spin,e--------------the spin and the electric charge of the black hole, 
!*                                     and -1 =< a_spin^2+e^2 <= 1 must be satisfied. 
!*               lambda,q,mve-------constants of motion, defined by lambda=L_z/E, q=Q/E^2, mve = \mu_m/E.
!*     OUTPUTS:  mu_tp1, mu_tp2----the turning points, between which the theta motion of 
!*                                 the particle is confined, and mu_tp2 <= mu_tp1. 
!*               reals-------------number of real roots of equation \Theta_\mu(\mu)=0.
!*               mobseqmtp---------If mobseqmtp=.TRUE., then muobs equals to be one of the turning points. 
!*     ROUTINES CALLED: NONE
!*     ACCURACY:   Machine.    
!*     AUTHOR:     Yang & Wang (2012)  
!*     DATE WRITTEN:  4 Jan 2012
!*     REVISIONS: ********************************************************************************************** 
      implicit none
      Double precision kvecp,kvect,sin_ini,cos_ini,a_spin,lambda,q,mve,zero,one,two,four,&
             mu_tp1,mu_tp2,delta,mutemp,Bprime,muplus,muminus,r1,r2,r3,r4
      integer  reals
      logical :: mobseqmtp
      parameter (zero=0.D0,two=2.0D0,four=4.D0,one=1.D0)
 
        mobseqmtp=.false.
        If(a_spin .eq. zero .OR. mve*mve-one .EQ. zero)then 
            If(kvect.ne.zero)then
                mu_tp1=sqrt(q/(lambda*lambda+q))
                mu_tp2=-mu_tp1 
            else
                mu_tp1=abs(cos_ini)
                mu_tp2=-mu_tp1
                mobseqmtp=.true.
            endif
            reals=2    
        ELSE 
            If(lambda.ne.zero)then
                delta=(a_spin**two*(mve*mve-one)+lambda**two+q)**two-four*a_spin**two*(mve*mve-one)*q
                muminus = -dsqrt(delta)+sign(one,mve*mve-one)*(lambda**two+q+a_spin**two*(mve*mve-one)) 
                IF(mve*mve-one .GE. zero)THEN
                    mu_tp1 = dsqrt(muminus/two/abs(mve*mve-one))/dabs(a_spin)
                    mu_tp2 = -mu_tp1 
                    IF(kvect.eq.zero)THEN
                        IF(abs(cos_ini-mu_tp1) .LE. 1.D-4)THEN
                            mu_tp1 = abs(cos_ini) 
                            mu_tp2 = -mu_tp1
                        ELSE
                            mu_tp2 = cos_ini 
                            mu_tp1 = -mu_tp2  
                        ENDIF  
                        mobseqmtp=.TRUE.                      
                    ENDIF  
                    reals=2  
                ELSE
                    mu_tp1 = dsqrt( abs( dsqrt(delta)+sign(one,mve*mve-one)*(lambda**two+q+&
                                    a_spin**two*(mve*mve-one)) )/two/abs(mve*mve-one) )/abs(a_spin) 
                    IF(muminus .GE. zero .or. abs(muminus).le.1.D-11)THEN
                        mu_tp2 = dsqrt(abs(muminus)/two/abs(mve*mve-one))/dabs(a_spin)
                        IF(kvect .EQ. zero)THEN
                            IF(abs(abs(cos_ini)-mu_tp1) .LE. 1.D-4)THEN
                                mu_tp1=dabs(cos_ini)
                            ELSE
                                mu_tp2=dabs(cos_ini)
                            ENDIF
                            mobseqmtp=.TRUE.
                        ENDIF  
                        reals=4
                    ELSE
                        IF(kvect .NE. zero)THEN
                            mu_tp2 = -mu_tp1
                        ELSE 
                            mu_tp1 = dabs(cos_ini)
                            mu_tp2 = -mu_tp1
                            mobseqmtp=.TRUE.
                        ENDIF 
                        reals=2                        
                    ENDIF
                ENDIF  
            else  !IF lambda=0, then \Theta_\mu=(1-mu^2)*(q-mu^2*a^2*(m^2-1)), so
                  !IF q <=0, mu1=sqrt(-q/a) and mu2=1 or -1.
                IF(mve*mve-one .ge. zero)then
                    r1 = dsqrt(abs(q/(mve*mve-one)))/dabs(a_spin)
                    IF(r1 .GE. one)THEN
                        If(kvect.ne.zero)then 
                            mu_tp1=one
                            mu_tp2=-one
                        else
                            IF(abs(cos_ini) .NE. one)THEN 
                                write(*,*)'mutp(): offending case:---1.!'
                                stop 
                            ELSE
                                mu_tp1=one
                                mu_tp2=-one!a=B=zero.
                                mobseqmtp=.true.
                            ENDIF
                        endif
                        reals=2
                    ELSE
                        IF(kvect .NE. zero)THEN
                            mu_tp1=r1
                            mu_tp2=-r1 
                        ELSE
                            mu_tp1=abs(cos_ini)
                            mu_tp2=-mu_tp1
                            mobseqmtp=.TRUE.
                        ENDIF
                    ENDIF 
                    reals=2
                else
                    IF(q.GT.zero)THEN
                        IF(kvect.ne.zero)THEN
                            mu_tp1 = one
                            mu_tp2 = -one
                        ELSE
                            IF(abs(cos_ini) .NE. one)THEN 
                                write(*,*)'mutp(): offending case:---1.!'
                                stop 
                            ELSE
                                mu_tp1=one
                                mu_tp2=-one!a=B=zero.
                                mobseqmtp=.true.
                            ENDIF 
                        ENDIF 
                        reals=2
                    ELSE
                        r1 = dsqrt(abs(q/(mve*mve-one)))/dabs(a_spin) 
                        IF(kvect.ne.zero)THEN
                            mu_tp1 = one
                            mu_tp2 = r1 
                        ELSE
                            IF(abs(abs(cos_ini)-r1) .LE.1.D-4)THEN  
                                mu_tp1=one
                                mu_tp2=abs(cos_ini) 
                                mobseqmtp=.true.
                            ELSE
                                mu_tp1=abs(cos_ini)
                                mu_tp2=r1  
                                mobseqmtp=.true.
                            ENDIF 
                        ENDIF  
                        reals=4
                    ENDIF 
                endif 
            endif
        ENDIF
        If(abs(cos_ini).eq.one)mobseqmtp=.true.
        If(cos_ini.lt.zero.and.reals.eq.4)then
            mutemp=mu_tp1
            mu_tp1=-mu_tp2
            mu_tp2=-mutemp
        endif 
        return
      end subroutine mutp 

!********************************************************************************************
      Subroutine radiustp(kvecr,a_spin,e,r_ini,lambda,q,mve,ep,r_tp1,&
                           r_tp2,reals,r_ini_eq_rtp,indrhorizon,r1eqr2,cases,bb)
!********************************************************************************************
!*     PURPOSE: Returns the coordinates r_tp1 and r_tp2 of turning points of the radial motion, judges
!*                whether the initial radius r_ini is equal to one of the turning points, if 
!*                it is true, then r_ini_eq_rtp=.TRUE.. And if r_tp1 less or equal r_horizon,
!*                then indrhorizon=.TRUE. Where r_horizon is the radius of the event horizon.  
!*     INPUTS:   kvecr----------k_{r}, the initial r component of four-momentum of the particle measured 
!*                              under the LNRF, see equation (93) in Yang & Wang (2013), A&A.
!*               a_spin,e--------------the spin and the electric charge of the black hole, 
!*                                     and -1 =< a_spin^2+e^2 <= 1 must be satisfied. 
!*               r_ini-----------------the initial radial coordinate of the particle. 
!*               lambda,q,mve,ep-------constants of motion, defined by lambda=L_z/E, q=Q/E^2,
!*                                     mve = \mu_m/E, ep=epsilon/varepsilon. 
!*     OUTPUTS:  r_tp1, r_tp2----the turning points, between which the radial motion of 
!*                                 the particle is confined, and r_tp2 >= r_tp1.
!*               bb(1:4)----roots of equation R(r)=0.                
!*               reals------number of real roots of equation R(r)=0.
!*               r_ini_eq_rtp---If r_ini_eq_rtp=.TRUE., then robs equal to be one of turning points. 
!*               cases-------If r_tp2 = infinity, then cases=1, else cases=2.
!*               indrhorizon----if r_tp1 less or equals r_horizon, indrhorizon=.TRUE.. 
!*     ROUTINES CALLED: root4
!*     ACCURACY:   Machine.    
!*     AUTHOR:     Yang & Wang (2012)  
!*     DATE WRITTEN:  4 Jan 2012
!*     REVISIONS: ****************************************** 
      implicit none
      Double precision a_spin,r_ini,lambda,q,r_tp1,r_tp2,&
             zero,one,two,four,a1,c1,d1,e1,r1(2),rhorizon,mve,kvecr,&
             ep,e,b0,b1,b2,b3,g2,g3,z_1,z_2,z_3,z_ini,z_tp1,z_tp2
      integer  reals,i,j,cases,del
      logical :: r_ini_eq_rtp,indrhorizon, r1eqr2
      complex*16 bb(1:4),r3(1:3),z3(1:3)
      parameter (zero=0.D0,two=2.0D0,four=4.D0,one=1.D0)  

      rhorizon=one+sqrt(one-a_spin*a_spin-e*e)
      r_ini_eq_rtp=.false.
      indrhorizon=.false.
      r1eqr2=.false.
      a1=one-mve*mve
      b1=two*(mve*mve+e*ep)/a1
      c1=-(a_spin*a_spin*(mve*mve-one)+lambda*lambda+q+e*e*(mve*mve-ep*ep))/a1
      d1=two*(q+(lambda-a_spin)*(lambda-a_spin)+a_spin*e*ep*(a_spin-lambda))/a1
      e1=-q*(a_spin*a_spin+e*e)/a1-e*e*(a_spin-lambda)**two/a1  
 
      IF(a1 .GT. zero)THEN
          call root4(b1,c1,d1,e1,bb(1),bb(2),bb(3),bb(4),reals)   
          SELECT CASE(reals) 
          CASE(4)  
              IF(kvecr.ne.zero)THEN
                If( r_ini.ge.real(bb(4)) )then 
                  r_tp1=real(bb(4)) 
                  r_tp2=infinity
                  cases=1 
                else
                  If( (r_ini.ge.real(bb(2)) .and. r_ini.le.real(bb(3))) )then 
                      r_tp1=real(bb(2)) 
                      r_tp2=real(bb(3))
                      cases=2
                  else
                      IF( real(bb(1)) .GT. rhorizon .AND.  r_ini .LE. real(bb(1))  )THEN
                          write(*,*)'radiustp(): wrong! 4 roots,cases = 3'
                          stop 
                          r_tp2=real(bb(1))  
                          r_tp1=-infinity 
                      ELSE   
                          write(*,*)bb,r_ini,kvecr
                          write(*,*)'radiustp(): wrong! 4 roots  ssssssss'
                          stop
                      ENDIF
                  endif
                endif 
              endif
              IF(kvecr.eq.zero)THEN
                  IF(dabs(r_ini-real(bb(4))) .LE. 1.D-4)THEN
                      r_tp1=r_ini
                      r_tp2=infinity
                      cases=1
                  ENDIF 
                  IF(dabs(r_ini-real(bb(2))) .LE. 1.D-4)THEN
                      r_tp1=r_ini  
                      r_tp2=real(bb(3))
                      cases=2
                  ENDIF
                  IF(dabs(r_ini-real(bb(3))) .LE. 1.D-4)THEN
                      r_tp1=real(bb(2))  
                      r_tp2=r_ini
                      cases=2
                  ENDIF
                  IF(dabs(r_ini-real(bb(1))) .LE. 1.D-4)THEN
                      r_tp1=-infinity
                      r_tp2=r_ini 
                      cases=3
                      write(*,*)'radiustp(): wrong! 4 roots, cases = 3'
                      stop  
                  ENDIF  
                  r_ini_eq_rtp = .TRUE. 
              ENDIF          
          CASE(2)
              j=1
              Do  i=1,4
                  If (aimag(bb(i)).eq.zero) then
                      r1(j)=real(bb(i))
                      j=j+1 
                  endif
              Enddo
              IF(kvecr.ne.zero)THEN
                If( r_ini.ge.r1(2) )then 
                  r_tp1=r1(2) 
                  r_tp2=infinity
                  cases=1
                else  
                  If( r1(1).ge.rhorizon .and. r_ini.le.r1(1) )then
                      write(*,*)'radiustp(): wrong! 2 roots, cases = 3'
                      stop 
                  endif
                endif
              Endif
              IF(kvecr.eq.zero)THEN 
                  IF(dabs(r_ini-r1(2)) .LE. 1.D-4)THEN
                      r_tp1=r_ini 
                      r_tp2=infinity
                  ENDIF
                  IF(dabs(r_ini-r1(1)) .LE. 1.D-4)THEN
                      r_tp1=-infinity 
                      r_tp2=r_ini 
                      write(*,*)'radiustp(): wrong! 2 roots, cases = 3'
                      stop
                  ENDIF
                  r_ini_eq_rtp=.TRUE. 
              ENDIF
          CASE(0)
              r_tp1=zero
              r_tp2=infinity
              cases=1  
          END SELECT 
      ELSE
      !***********************
        IF(a1 .LT. zero)THEN
          call root4(b1,c1,d1,e1,bb(1),bb(2),bb(3),bb(4),reals)   
          SELECT CASE(reals)
          CASE(4)
              IF(kvecr.ne.zero)THEN 
                If( r_ini.LE.real(bb(4)) .AND. r_ini.GE.real(bb(3)) )then 
                  r_tp1=real(bb(3)) 
                  r_tp2=real(bb(4))
                  cases=2
                else
                  If( r_ini.ge.real(bb(1)) .and. r_ini.le.real(bb(2)) )then 
                      r_tp1=real(bb(1)) 
                      r_tp2=real(bb(2))
                      cases=2
                  endif
                endif
              endif
              IF(kvecr.eq.zero)THEN
                  IF(dabs(r_ini-real(bb(3))) .LE. 1.D-4)THEN
                      r_tp1=r_ini 
                      r_tp2=real(bb(4))
                  ENDIF 
                  IF(dabs(r_ini-real(bb(4))) .LE. 1.D-4)THEN
                      r_tp1=real(bb(3))  
                      r_tp2=r_ini
                  ENDIF 
                  IF(dabs(r_ini-real(bb(1))) .LE. 1.D-4)THEN
                      r_tp1=r_ini  
                      r_tp2=real(bb(2))
                  ENDIF
                  IF(dabs(r_ini-real(bb(2))) .LE. 1.D-4)THEN
                      r_tp1=real(bb(1))  
                      r_tp2=r_ini 
                  ENDIF 
                  cases=2
                  r_ini_eq_rtp=.TRUE.  
              ENDIF           
          CASE(2)
              j=1
              Do  i=1,4 
                  If (aimag(bb(i)).eq.zero) then
                      r1(j)=real(bb(i)) 
                      j=j+1 
                  endif
              Enddo  
              IF(kvecr.ne.zero)THEN
                  If( r_ini.le.r1(2) .AND. r_ini.GE.r1(1) )then
                      r_tp1=r1(1) 
                      r_tp2=r1(2)
                  ENDIF   
              Endif         
              IF( kvecr.eq.zero )THEN
                  IF(dabs(r_ini-r1(1)) .LE. 1.D-4)THEN
                      r_tp1=r_ini 
                      r_tp2=r1(2) 
                  ENDIF
                  IF(dabs(r_ini-r1(2)) .LE. 1.D-4)THEN
                      r_tp1=r1(1)
                      r_tp2=r_ini 
                  ENDIF
                  r_ini_eq_rtp=.TRUE.  
              endif
              cases=2 
          CASE(0)
              write(*,*)'radiustp(): wrong! 1-m^2 <0, no real roots exist, which',&
                        'can not happen.'
              stop   
          END SELECT 
        ELSE  !mve=1, R(r) becomes cubic polynomials directly.
          b1=two*(one+e*ep) 
          c1=-(lambda*lambda+q+e*e*(one-ep*ep))/b1 
          d1=two*(q+(lambda-a_spin)*(lambda-a_spin)+a_spin*e*ep*(a_spin-lambda))/b1 
          e1=-q*(a_spin*a_spin+e*e)/b1-e*e*(a_spin-lambda)**two/b1 

          call root3(c1,d1,e1,r3(1),r3(2),r3(3),reals) 

          SELECT CASE(reals)
          CASE(3)
              IF( b1 .gt. zero )THEN
                  IF(kvecr.ne.zero)THEN 
                      IF( r_ini .GE. real(r3(1)) )THEN
                          r_tp1=real(r3(1))
                          r_tp2=infinity
                          cases=1
                      ELSE
                          IF( r_ini.le.real(r3(2)) .AND. r_ini.ge.real(r3(3)) )THEN
                              r_tp1=real(r3(3))
                              r_tp2=real(r3(2))
                              cases=2           
                          ENDIF
                      ENDIF 
                  Endif
                  IF(kvecr.eq.zero)THEN
                      IF(abs(r_ini-real(r3(1))) .LE. 1.D-4)THEN
                          r_tp1=r_ini
                          r_tp2=infinity
                          cases=1
                      ENDIF
                      IF(abs(r_ini-real(r3(2))) .LE. 1.D-4)THEN
                          r_tp2=r_ini 
                          r_tp1=real(r3(3))
                          cases=2
                      ENDIF
                      IF(abs(r_ini-real(r3(3))) .LE. 1.D-4)THEN
                          r_tp1=r_ini 
                          r_tp2=real(r3(2))
                          cases=2
                      ENDIF 
                      r_ini_eq_rtp=.TRUE.  
                  ENDIF 
              ELSE
                  IF(kvecr.ne.zero)THEN 
                      IF( r_ini .lE. real(r3(3)) )THEN
                          r_tp2=real(r3(3))
                          r_tp1=-infinity
                          cases=3
                      ELSE
                          IF( r_ini.le.real(r3(1)) .AND. r_ini.ge.real(r3(2)) )THEN
                              r_tp1=real(r3(2))
                              r_tp2=real(r3(1))
                              cases=2           
                          ENDIF
                      ENDIF 
                  Endif
                  IF(kvecr.eq.zero)THEN
                      IF(abs(r_ini-real(r3(1))) .LE. 1.D-4)THEN
                          r_tp2=r_ini
                          r_tp1=real(r3(2))
                          cases=2
                      ENDIF
                      IF(abs(r_ini-real(r3(2))) .LE. 1.D-4)THEN
                          r_tp1=r_ini 
                          r_tp2=real(r3(1))
                          cases=2
                      ENDIF
                      IF(abs(r_ini-real(r3(3))) .LE. 1.D-4)THEN
                          r_tp1=-infinity
                          r_tp2=real(r3(3))
                          cases=3
                      ENDIF 
                      r_ini_eq_rtp=.TRUE.  
                  ENDIF                   
              ENDIF
          CASE(1) 
              IF(b1 .ge. zero)THEN
                  IF( r_ini .GE. real(r3(1)) )THEN
                      r_tp1=real(r3(1))
                      r_tp2=infinity
                      cases=1 
                  ELSEIF( kvecr.eq.zero )THEN 
                      r_tp1=r_ini
                      r_tp2=infinity
                      cases=1 
                      r_ini_eq_rtp=.TRUE.   
                  ELSE
                      write(*,*)'radiustp(): wrong! mve=1, and R(r)=0 has one',&
                           'real root r1, but r_ini<r1, this can not be happen!'
                  ENDIF 
              ELSE
                  IF(kvecr .ne. zero)THEN
                      IF( r_ini .LT. real(r3(1)) )THEN
                          r_tp1 = -infinity
                          r_tp2 = real(r3(1))
                          cases = 3
                      ENDIF
                  ENDIF
                  IF(kvecr .eq. zero)THEN
                      IF( r_ini .LT. real(r3(1)) )THEN
                          r_tp1 = -infinity
                          r_tp2 = r_ini
                          cases = 3
                      ENDIF
                      r_ini_eq_rtp=.TRUE.  
                  ENDIF
              ENDIF 
          END SELECT
!============================================================
        ENDIF    
      !***********************
      ENDIF   
 
      IF(rhorizon.ge.r_tp1 .and. rhorizon.le.r_tp2)then
          indrhorizon=.true.
      Endif    
      RETURN
      End Subroutine radiustp 


!******************************************************************************************** 
      Function mu2p(kp,kt,lambda,q,mve,mu,sin_ini,cos_ini,a_spin,t1,t2)
!********************************************************************************************
!*     PURPOSE:  Computes the value of parameter p from \mu coordinate. In other words, to compute 
!*               the \mu part of integral of equation (23) p=-sign(p_\theta)*p_0+2*t1*p_2+2*t2*p_2. 
!*               where p_\theta is initial \theta component of 4 momentum of photon.
!*     INPUTS:   kt--------k_{theta} or p_\theta, which is the \theta component of four momentum of a photon 
!*                              measured under the LNRF, see equation (94) in Yang & Wang (2013).
!*               kp--------k_{phi} or p_\phi, which is the \phi component of four momentum of a photon 
!*                              measured under the LNRF, see equation (95) in Yang & Wang (2013).
!*               lambda,q,mve-------constants of motion, defined by lambda=L_z/E, q=Q/E^2, mve = \mu_m/E.
!*               sin_ini,cos_ini-------sin_ini=sin(\theta_{ini}), cos_ini=cos(\theta_{ini}), where 
!*                                     \theta_{ini} is the initial \theta coordinate of the particle.
!*               a_spin---------spin of black hole, on interval [-1,1].
!*               t1,t2----------Number of photon meets the turning points \mu_tp1 and \mu_tp2
!*                              respectively.    
!*               mu-------------\mu coordinate of the particle.     
!*     OUTPUTS:  value of \mu part of integral of (23). 
!*     ROUTINES CALLED: mu2p_schwartz, mutp, root3, weierstrass_int_J3
!*     ACCURACY:   Machine.    
!*     AUTHOR:     Yang & Wang (2012)  
!*     DATE WRITTEN:  4 Jan 2012
!*     REVISIONS: ****************************************** 
      implicit none
      Double precision mu2p,kp,kt,mu,sin_ini,cos_ini,a_spin,lambda,q,mve,mu_tp1,tposition,&
                    tp2,four,bc,cc,dc,ec,b0,b1,b2,b3,g2,g3,tobs,p1,p2,p0,a4,b4,&
                    delta,two,mu_tp2,zero,mutemp,one,integ5(5),three,rff_p,come
      parameter (zero=0.D0,two=2.D0,four=4.D0,one=1.D0,three=3.D0)
      integer  t1,t2,reals,index_p5(5),del,cases
      complex*16 bb(1:4),dd(3)
      logical :: mobseqmtp !this veriable to check if cos_ini eq mu_tp(or mu_tp2)if then it's true!

      If(kp.eq.zero .and. kt.eq.zero .and. abs(cos_ini).eq.one)then
          IF(mve .LE. one)THEN    !In this case we have lambda=0, q=a^2(m^2-1)                
              mu2p=zero!          !so \Theta=(1-mu^2)(q-mu^2a^2(m^2-1))=a^2(m^2-1)(1-mu^2)^2
              return              !so if m>1 mu E (-1,1), if m<1 \Theta<0, so mu==+1 or -1.         
          ENDIF                   !if m=1,then \Theta_ini=0, so \theta_dot=0, so mu remains to
      endif                       !be +1 or -1. 
      If(a_spin.eq.zero .OR. mve-one.eq.zero)then
          call mu2p_schwarz(kp,kt,lambda,q,mve,mu,sin_ini,cos_ini,a_spin,t1,t2,mu2p)
          return 
      endif
       
      a4=zero
      b4=one 
      mobseqmtp=.false.
      call mutp(kp,kt,sin_ini,cos_ini,a_spin,lambda,q,mve,mu_tp1,mu_tp2,reals,mobseqmtp)
      If(mu_tp1.eq.zero)then
          mu2p=zero
          return
      endif 

      If(mu.gt.mu_tp1.or.mu.lt.mu_tp2)then
          mu2p=-one
          return
      endif 
      come = mve*mve-one  
      b0=four*a_spin**2*mu_tp1**3*come-two*mu_tp1*(a_spin**2*come+lambda**2+q)
      b1=two*a_spin**2*mu_tp1**2*come-(a_spin**2*come+lambda**2+q)/three
      b2=four/three*a_spin**2*mu_tp1*come
      b3=a_spin**2*come
      g2=three/four*(b1**2-b0*b2)
      g3=(three*b0*b1*b2-two*b1**3-b0**2*b3)/16.D0
 
      If(abs(mu-mu_tp1).ne.zero)then
          tposition=b0/(four*(mu-mu_tp1))+b1/four
      else
          tposition=infinity 
      endif
      If(cos_ini.ne.mu_tp1)then 
          tobs=b0/four/(cos_ini-mu_tp1)+b1/four
      else
          tobs=infinity
      endif   
      tp2=b0/four/(mu_tp2-mu_tp1)+b1/four  
      call root3(zero,-g2/four,-g3/four,dd(1),dd(2),dd(3),del) 

      index_p5(1)=0
      cases=1  
      call weierstrass_int_J3(tobs,tposition,dd,del,a4,b4,index_p5,rff_p,integ5,cases)  
      p0=integ5(1) 
      If(t1.eq.0)then
          p1=zero
      else 
          call weierstrass_int_J3(tposition,infinity,dd,del,a4,b4,index_p5,rff_p,integ5,cases)
          p1=integ5(1)
      endif
      If(t2.eq.0)then
          p2=zero
      else
          call weierstrass_int_J3(tp2,tposition,dd,del,a4,b4,index_p5,rff_p,integ5,cases) 
          p2=integ5(1)
      endif

      If(mobseqmtp)then  !acturally kt equal to be zero.
          !If(cos_ini.eq.mu_tp1)then  
          !    mu2p=-p0+two*(t1*p1+t2*p2)  
          !else
          !    mu2p=p0+two*(t1*p1+t2*p2)  
          !endif 
          mu2p=abs(p0)+two*(t1*p1+t2*p2)   
      else
          If(kt.gt.zero)then
              mu2p=-p0+two*(t1*p1+t2*p2)
          endif
          If(kt.lt.zero)then  
              mu2p=p0+two*(t1*p1+t2*p2)
          endif 
      endif
      return
      end Function mu2p


!********************************************************************************************
      subroutine mu2p_schwarz(kp,kt,lambda,q,mve,mu,sin_ini,cos_ini,a_spin,t1,t2,mu2p)
!********************************************************************************************
!*     PURPOSE:  Computes the value of parameter p from \mu coordinate. In other words, to compute 
!*               the \mu part of integral of equation (23) p=-sign(p_\theta)*p_0+2*t1*p_2+2*t2*p_2. 
!*               where p_\theta is initial \theta component of 4 momentum of photon.
!*               And the black hole spin a_spin = 0.
!*     INPUTS:   kt--------k_{theta} or p_\theta, which is the \theta component of four momentum of a photon 
!*                              measured under the LNRF, see equation (94) in Yang & Wang (2013).
!*               kp--------k_{phi} or p_\phi, which is the \phi component of four momentum of a photon 
!*                              measured under the LNRF, see equation (95) in Yang & Wang (2013).
!*               lambda,q,mve-------constants of motion, defined by lambda=L_z/E, q=Q/E^2, mve = \mu_m/E.
!*               sin_ini,cos_ini-------sin_ini=sin(\theta_{ini}), cos_ini=cos(\theta_{ini}), where 
!*                                     \theta_{ini} is the initial \theta coordinate of the particle.
!*               a_spin---------spin of black hole, on interval [-1,1].
!*               t1,t2----------Number of photon meets the turning points \mu_tp1 and \mu_tp2
!*                              respectively.    
!*               mu-------------\mu coordinate of the particle.     
!*     OUTPUTS:  mu2p-----------value of \mu part of integral of (23). 
!*     ROUTINES CALLED:  
!*     ACCURACY:   Machine.    
!*     AUTHOR:     Yang & Wang (2012)  
!*     DATE WRITTEN:  4 Jan 2012
!*     REVISIONS: ****************************************** 
      implicit none
      Double precision kp,kt,mu,sin_ini,cos_ini,mu2p,pp,p1,p2,AA,BB,two,DD,&
                     lambda,q,a_spin,zero,one,mu_tp,mu_tp2,mve
      integer  t1,t2 
      parameter(two=2.D0,zero=0.D0,one=1.D0)
      logical :: mobseqmtp

      If(kp.eq.zero .and. kt.eq.zero)then !in this case lambda=q=0, so \Theta=q(1-mu^2)=0 for ever, 
          mu2p=-two                       !we do not need to consider it.
          return                          !it will return
      endif                               !zero value.
      mobseqmtp=.false. 
      If(q.gt.zero)then 
          BB=sqrt(q)
          If(kt.ne.zero)then
              mu_tp=sqrt(q/(lambda**two+q))
              mu_tp2=-mu_tp 
          else
              mu_tp=cos_ini
              mu_tp2=-mu_tp
              mobseqmtp=.true.
          endif
          !If(abs(mu).gt.mu_tp)then
          !   mu2p=-one
          !   return
          !endif
          If(abs(cos_ini).eq.one)mobseqmtp=.true.  
          pp=(asin(mu/mu_tp)-asin(cos_ini/mu_tp))*mu_tp/BB 
          If(t1.eq.0)then
              p1=zero
          else 
              p1=(PI/two-asin(mu/mu_tp))*mu_tp/BB 
          endif
          If(t2.eq.0)then
              p2=zero
          else 
              p2=(asin(mu/mu_tp)+PI/two)*mu_tp/BB
          endif
          If(mobseqmtp)then
              !If(cos_ini.eq.mu_tp)then  
              !   mu2p=-pp+two*(t1*p1+t2*p2)  
              !else
              !    mu2p=pp+two*(t1*p1+t2*p2)  
              !endif
              mu2p=abs(pp)+two*(t1*p1+t2*p2)   
          else
              mu2p=sign(one,-kt)*pp+two*(t1*p1+t2*p2)  
          endif
      else 
          mu2p=zero
      endif  
      return
      end subroutine mu2p_schwarz


!********************************************************************************************
      Function r2p(kr,rend,lambda,q,mve,ep,a_spin,e,r_ini,t1,t2)
!============================================================================================
!*     PURPOSE:  Computes the value of parameter p from radial coordinate rend. In other words, to compute 
!*               the r part of integral of equation (23), using formula: 
!*               p=-sign(p_r)*p_0+2*t1*p_2+2*t2*p_2. where p_r is initial radial
!*               component of 4 momentum of photon. 
!*     INPUTS:   kr--------------------k_{r}.
!*               rend------------------the end radial coordinate.
!*               lambda,q,mve,ep-------constants of motion, defined by lambda=L_z/E, q=Q/E^2,
!*                                     mve = \mu_m/E, ep=epsilon/varepsilon. 
!*               a_spin,e--------------the spin and the electric charge of the black hole, 
!*                                     and -1 =< a_spin^2+e^2 <= 1 must be satisfied. 
!*               r_ini-----------------the initial radial coordinate of the particle.   
!*               t1,t2-----------------Number of photon meets the turning points r_tp1 and r_tp2
!*                                     respectively in radial motion.
!*     OUTPUTS:  r2p-------------------value of r part of integral (23) in Yang & Wang (2013), A&A.
!*     ROUTINES CALLED: radiustp, root3, weierstrass_int_j3, EllipticF.
!*     ACCURACY:   Machine.    
!*     AUTHOR:     Yang & Wang (2012)  
!*     DATE WRITTEN:  4 Jan 2012
!*     REVISIONS: ****************************************** 
      implicit none
      Double precision r2p,p,a_spin,e,rhorizon,q,lambda,zero,integ5(5),&
             bc,dc,ec,b0,b1,b2,b3,g2,g3,tobs,tinf1,PI1,r_ini,cr,dr,integ05(5),&
             u,v,w,s,L1,L2,thorizon,m2,pinf,sn,cn,dn,a4,b4,one,two,four,PI2,ttp,sqrt3,&
             integ14(4),three,six,nine,r_tp1,r_tp2,kr,tp2,tp,t_inf,ac1,bc1,cc1,&
             p0,p1,p2,rend,rff_p,mve,ep,come,rinf,tp1,alpha1,alpha2,ar(5) 
      parameter(zero=0.D0,one=1.D0,two=2.D0,four=4.D0,three=3.D0,six=6.D0,nine=9.D0)
      complex*16 bb(1:4),dd(3)
      integer  reals,i,p4,cases_int,del,index_p5(5),cases,t1,t2
      logical :: r_ini_eq_rtp,indrhorizon,r1eqr2
 
      rhorizon=one+sqrt(one-a_spin*a_spin-e*e)
      a4=zero
      b4=one 
      r_ini_eq_rtp=.false.
      indrhorizon=.false.
 
      IF(mve .EQ. one)THEN
          r2p=r2p_mb(kr,rend,lambda,q,mve,ep,a_spin,e,r_ini,t1,t2) 
          RETURN
      ENDIF 
      call radiustp(kr,a_spin,e,r_ini,lambda,q,mve,ep,r_tp1,&
                     r_tp2,reals,r_ini_eq_rtp,indrhorizon,r1eqr2,cases,bb) 

      !write(*,*)'reals=',reals,rend,r_tp1,r_tp2
      IF(r1eqr2)THEN
          r2p=zero
          return
      ENDIF
      come = mve*mve-one 
      If(reals.ne.0)then
          If((rend-r_tp1)*(rend-r_tp2) .GT. zero)then
              r2p=-one
              return
          endif
          ar(1)=-come 
          ar(2)=two*(mve*mve+e*ep) 
          ar(3)=-(a_spin*a_spin*come+lambda*lambda+q+e*e*(mve*mve-ep*ep)) 
          ar(4)=two*(q+(lambda-a_spin)*(lambda-a_spin)+a_spin*e*ep*(a_spin-lambda)) 
          ar(5)=-q*(a_spin*a_spin+e*e)-e*e*(a_spin-lambda)**two 

          b0 = four*r_tp1**three*ar(1)+three*ar(2)*r_tp1**two+two*ar(3)*r_tp1+ar(4)
          b1 = two*r_tp1**two*ar(1)+two*ar(2)*r_tp1+ar(3)/three
          b2 = four/three*r_tp1*ar(1)+ar(2)/three
          b3 = ar(1)
          g2 = three/four*(b1*b1-b0*b2)
          g3 = (three*b0*b1*b2-two*b1*b1*b1-b0*b0*b3)/16.D0
          rinf = r_tp1  
          tp1 = infinity
          If(r_ini-rinf.ne.zero)then 
              tobs=b0/four/(r_ini-rinf)+b1/four
          else
              tobs=infinity
          endif 
          If(rhorizon-rinf.ne.zero)then
              thorizon=b1/four+b0/four/(rhorizon-rinf)
          else
              thorizon=infinity  
          endif
          If(rend-rinf.ne.zero)then
              tp=b1/four+b0/four/(rend-rinf)
          else
              tp=infinity  
          endif
          tp2=b0/four/(r_tp2-rinf)+b1/four      
 
          call root3(zero,-g2/four,-g3/four,dd(1),dd(2),dd(3),del) 
          index_p5(1)=0 
          cases_int=1 
 
          call weierstrass_int_J3(tobs,tp,dd,del,a4,b4,index_p5,rff_p,integ5,cases_int)
          p0=integ5(1) 
          If(t1.eq.zero)then
              p1=zero
          else
              call weierstrass_int_J3(tp,tp1,dd,del,a4,b4,index_p5,rff_p,integ5,cases_int)
              p1=integ5(1) 
          endif 
          If(t2.eq.zero)then
              p2=zero
          else
              call weierstrass_int_J3(tp2,tp,dd,del,a4,b4,index_p5,rff_p,integ5,cases_int)
              p2=integ5(1) 
          endif
          If( .not. r_ini_eq_rtp )then
              r2p=sign(one,-kr)*p0+two*(t1*p1+t2*p2)
          else 
              r2p=abs(p0)+two*(t1*p1+t2*p2)  
          endif   
      else
          IF(-come .GT. zero)THEN
              u=real(bb(4))
              w=abs(aimag(bb(4)))
              v=real(bb(3))
              s=abs(aimag(bb(3)))
              ac1=s*s
              bc1=w*w+s*s+(u-v)*(u-v)
              cc1=w*w 
              L1=(bc1+dsqrt(bc1*bc1-four*ac1*cc1))/two/ac1            
              L2=(bc1-dsqrt(bc1*bc1-four*ac1*cc1))/two/ac1 
              alpha1=(L1*v-u)/(L1-one)
              alpha2=(L2*v-u)/(L2-one)
              thorizon = dsqrt((L1-one)/(L1-L2))*(rhorizon-alpha1)/dsqrt((rhorizon-v)**2+ac1)  
              tobs = dsqrt((L1-one)/(L1-L2))*(r_ini-alpha1)/dsqrt((r_ini-v)**2+ac1)  
              tp = dsqrt((L1-one)/(L1-L2))*(rend-alpha1)/dsqrt((rend-v)**2+ac1) 
              t_inf = dsqrt((L1-one)/(L1-L2))
              m2 = (L1-L2)/L1      
              pinf = EllipticF(tobs,m2) 
              If(kr.lt.zero)then
                  If(rend.le.rhorizon)then  
                      tp=thorizon                  
                  endif           
                  r2p = ( pinf-EllipticF(tp,m2) )/s/sqrt(-come*L1)      
              else
                  If(rend.GE.infinity)then
                      tp = t_inf
                  endif 
                  r2p = ( EllipticF(tp,m2)-pinf )/s/sqrt(-come*L1) 
              endif
          ENDIF           
          IF(-come .LT. zero)THEN
              u=real(bb(4))
              w=abs(aimag(bb(4)))
              v=real(bb(3))
              s=abs(aimag(bb(3)))
              ac1=s*s
              bc1=-w*w-s*s-(u-v)*(u-v)
              cc1=w*w 
              L1=(bc1+dsqrt(bc1*bc1-four*ac1*cc1))/two/ac1            
              L2=(bc1-dsqrt(bc1*bc1-four*ac1*cc1))/two/ac1 
              alpha1=(L1*v+u)/(L1+one)
              alpha2=(L2*v+u)/(L2+one)  
              thorizon = dsqrt( (L1-L2)/(-L2*(one+L1)) )*(rhorizon-alpha1)/dsqrt( (rhorizon-u)**2+cc1 ) 
              tobs = dsqrt( (L1-L2)/(-L2*(one+L1)) )*(r_ini-alpha1)/dsqrt( (r_ini-u)**2+cc1 ) 
              tp = dsqrt( (L1-L2)/(-L2*(one+L1)) )*(rend-alpha1)/dsqrt( (rend-u)**2+cc1 ) 
              t_inf = dsqrt( (L1-L2)/(-L2*(one+L1)) ) 
              m2 = (L2-L1)/L2
              pinf = EllipticF(tobs,m2)
              If(kr.lt.zero)then
                  If(rend.le.rhorizon)then  
                      tp=thorizon                  
                  endif           
                  r2p = ( pinf-EllipticF(tp,m2) )/s/sqrt(-come*L2)      
              else
                  If(rend.GE.infinity)then
                      tp = t_inf
                  endif 
                  r2p = ( EllipticF(tp,m2)-pinf )/s/sqrt(-come*L2) 
              endif               
          ENDIF       
      endif 
      return   
      End function r2p


!********************************************************************************************
      FUNCTION r2p_mb(kr,rend,lambda,q,mve,ep,a_spin,e,r_ini,t1,t2) 
!============================================================================================
!*     PURPOSE:  Computes the value of parameter p from radial coordinate rend. In other words, to compute 
!*               the r part of integral of equation (23), using formula: 
!*               p=-sign(p_r)*p_0+2*t1*p_2+2*t2*p_2. where p_r is initial radial
!*               component of 4 momentum of photon. 
!*               And the constant mve = 1.
!*     INPUTS:   kr--------------------k_{r}.
!*               rend------------------the end radial coordinate.
!*               lambda,q,mve,ep-------constants of motion, defined by lambda=L_z/E, q=Q/E^2,
!*                                     mve = \mu_m/E, ep=epsilon/varepsilon. 
!*               a_spin,e--------------the spin and the electric charge of the black hole, 
!*                                     and -1 =< a_spin^2+e^2 <= 1 must be satisfied. 
!*               r_ini-----------------the initial radial coordinate of the particle.   
!*               t1,t2-----------------Number of photon meets the turning points r_tp1 and r_tp2
!*                                     respectively in radial motion.
!*     OUTPUTS:  r2p-------------------value of r part of integral (23) in Yang & Wang (2013), A&A.
!*     ROUTINES CALLED: radiustp, root3, weierstrass_int_j3, EllipticF.
!*     ACCURACY:   Machine.    
!*     AUTHOR:     Yang & Wang (2012)  
!*     DATE WRITTEN:  4 Jan 2012
!*     REVISIONS: ****************************************** 
      USE constants 
      IMPLICIT NONE
      Double precision r2p_mb,p,a_spin,e,rhorizon,q,lambda,integ5(5),&
             bc,dc,ec,b0,b1,b2,b3,g2,g3,tobs,tinf1,PI1,r_ini,cr,dr,integ05(5),&
             u,v,w,s,L1,L2,thorizon,m2,pinf,sn,cn,dn,a4,b4,PI2,ttp,sqrt3,&
             integ14(4),r_tp1,r_tp2,kr,tp2,tp,t_inf,ac1,bc1,cc1,sqrtb0,&
             p0,p1,p2,rend,rff_p,mve,ep,come,rinf,tp1,alpha1,alpha2  
      complex*16 bb(1:4),dd(3),ddr(3)
      integer  reals,i,p4,cases_int,del,delr,index_p5(5),cases,t1,t2
      logical :: r_ini_eq_rtp,indrhorizon,r1eqr2
 
      rhorizon=one+sqrt(one-a_spin*a_spin-e*e)
      a4=zero
      b4=one 
      r_ini_eq_rtp=.false.
      indrhorizon=.false.
      call radiustp(kr,a_spin,e,r_ini,lambda,q,mve,ep,r_tp1,&
                     r_tp2,reals,r_ini_eq_rtp,indrhorizon,r1eqr2,cases,bb) 
 
      b0=two*(one+e*ep) ! we assume |e*ep|<1, thus b0>0
      b1=-(lambda*lambda+q+e*e*(one-ep*ep))/three
      b2=two*(q+(lambda-a_spin)*(lambda-a_spin)+a_spin*e*ep*(a_spin-lambda))/three
      b3=-q*(a_spin*a_spin+e*e)-e*e*(a_spin-lambda)**two
 
      g2 = three/four*(b1*b1-b0*b2)
      g3 = (three*b0*b1*b2-two*b1*b1*b1-b0*b0*b3)/16.D0  
 
      call root3(b1/b0,b2/b0,b3/b0,dd(1),dd(2),dd(3),del) !dd for integral computing. 
      !call root3(zero,-g2/four,-g3/four,ddr(1),ddr(2),ddr(3),delr) !ddr for compute coordinates. 
 
      tp1 = b0/four*r_tp1+b1/four
      IF(r_tp2.LT.infinity)THEN
          tp2 = b0/four*r_tp2+b1/four   
      ELSE
          tp2 = infinity
      ENDIF  
      tobs = b0/four*r_ini+b1/four 
      tp = b0/four*rend+b1/four
      index_p5(1)=0 
      cases_int=1 
 
      call weierstrass_int_J3(tobs,tp,dd,del,a4,b4,index_p5,rff_p,integ5,cases_int)
      p0=integ5(1) 

      If(t1.eq.zero)then
          p1=zero
      else
          call weierstrass_int_J3(tp1,tp,dd,del,a4,b4,index_p5,rff_p,integ5,cases_int)
          p1=integ5(1) 
      endif 
      If(t2.eq.zero)then
          p2=zero
      else
          call weierstrass_int_J3(tp,tp2,dd,del,a4,b4,index_p5,rff_p,integ5,cases_int)
          p2=integ5(1) 
      endif
      If( .not. r_ini_eq_rtp )then
          r2p_mb=sign(one,kr)*p0+two*(t1*p1+t2*p2)
      else 
          r2p_mb=abs(p0)+two*(t1*p1+t2*p2)  
      endif   
      RETURN      
      END FUNCTION r2p_mb  

 
!********************************************************************************************
      Function mucos(p,kp,kt,lambda,q,mve,sin_ini,cos_ini,a_spin)
!============================================================================================
!*     PURPOSE:  Computes function \mu(p) defined in Table 1 of Yang & Wang (2013), A&A. That is
!*               \mu(p)=b0/(4*\wp(p+PI0;g_2,g_3)-b1)+\mu_tp1. \wp(p+PI0;g_2,g_3) is the Weierstrass'
!*               elliptic function.  
!*     INPUTS:   p--------------independent variable, which must be nonnegative.
!*               kp-------------k_{\phi}, initial \phi component of four momentum of a particle
!*                              measured under an LNRF.
!*               kt-------------k_{\theta}, initial \theta component of four momentum of a particle
!*                              measured under an LNRF. 
!*               lambda,q,mve-------constants of motion, defined by lambda=L_z/E, q=Q/E^2, mve = \mu_m/E.
!*               sin_ini,cos_ini-------sin_ini=sin(\theta_{ini}), cos_ini=cos(\theta_{ini}), where 
!*                                     \theta_{ini} is the initial \theta coordinate of the particle.
!*               a_spin---------spin of the black hole, on interval [-1,1].        
!*     OUTPUTS:  mucos----------\mu coordinate of particle corresponding to a given p. 
!*     ROUTINES CALLED: weierstrass_int_J3, mutp, root3.
!*     ACCURACY:   Machine.    
!*     AUTHOR:     Yang & Wang (2012)  
!*     DATE WRITTEN:  4 Jan 2012
!*     REVISIONS: ****************************************** 
      implicit none
      Double precision mucos,kp,kt,p,sin_ini,cos_ini,a_spin,q,lambda,bc,cc,dc,ec,mu_tp1,&
               zero,b0,b1,b2,b3,g2,g3,tinf,PI0,a4,b4,AA,BB,two,delta,four,&
               mu_tp2,one,three,integ5(5),rff_p,mve,come
      Double precision kp_1,kt_1,lambda_1,q_1,sin_ini_1,cos_ini_1,a_spin_1,mve_1
      complex*16 dd(3)
      integer ::  reals,p4,index_p5(5),del,cases,count_num=1
      logical :: mobseqmtp
      save  kp_1,kt_1,lambda_1,q_1,sin_ini_1,cos_ini_1,a_spin_1,mve_1,&
                  mu_tp1,mu_tp2,b0,b1,b2,b3,g2,g3,dd,PI0,count_num,AA,BB
      parameter (zero=0.D0,two=2.0D0,four=4.D0,one=1.D0,three=3.D0)
 
      IF(p.lt.zero)THEN
          Write(*,*)'mucos(): p you given is <0, which is not valid and the code should be stopped.'
          Write(*,*)'Please input a positive one and try it again.'
          STOP
      ENDIF   
10    continue
      If(count_num.eq.1)then
          kp_1=kp
          kt_1=kt
          lambda_1=lambda
          q_1=q
          mve_1=mve
          cos_ini_1=cos_ini
          sin_ini_1=sin_ini
          a_spin_1=a_spin
     !*****************************************************************************************************
          If(kp.eq.zero .and. kt.eq.zero .and. abs(cos_ini).eq.one)then
              IF(mve .LE. one)THEN    !In this case we have lambda=0, q=a^2(m^2-1)                
                  mucos=cos_ini         !so \Theta=(1-mu^2)(q-mu^2a^2(m^2-1))=a^2(m^2-1)(1-mu^2)^2
                  return              !so if m>1 mu E (-1,1), if m<1 \Theta<0, so mu==+1 or -1.         
              ENDIF                   !if m=1,then \Theta_ini=0, so \theta_dot=0, so mu remains to
          endif                       !be +1 or -1.  
          If(a_spin.eq.zero .or. mve .eq. one)then
              if(q.gt.zero)then 
                  AA=dsqrt((lambda**two+q)/q)
                  BB=dsqrt(q)
                  If(kt.gt.zero)then
                      mucos=sin(asin(cos_ini*AA)-p*BB*AA)/AA 
                  else
                      If(kt.lt.zero)then
                          mucos=sin(asin(cos_ini*AA)+p*AA*BB)/AA
                      else
                          mucos=cos(p*AA*BB)*cos_ini !AA=1/cos_ini,sin(asin(cos_ini/cos_ini)+/-p*AA*BB)/AA          
                      endif                   !=sin(Pi/2+/-p*AA*BB)*cos_ini=cos(p*AA*BB)*cos_ini.  
                  endif
              else
                  mucos=cos_ini
              endif
          else
              If(cos_ini.eq.zero.and.q.eq.zero)then
                  mucos=zero
                  return   
              endif
              call mutp(kp,kt,sin_ini,cos_ini,a_spin,lambda,q,mve,mu_tp1,mu_tp2,reals,mobseqmtp) 
              a4=zero
              b4=one
              p4=0
              come=mve*mve-one
              b0=four*a_spin**2*mu_tp1**3*come-two*mu_tp1*(a_spin**2*come+lambda**2+q)
              b1=two*a_spin**2*mu_tp1**2*come-(a_spin**2*come+lambda**2+q)/three
              b2=four/three*a_spin**2*come*mu_tp1
              b3=a_spin**2*come
              g2=three/four*(b1**2-b0*b2)
              g3=one/16.D0*(three*b0*b1*b2-two*b1**3-b0**2*b3)

              call root3(zero,-g2/four,-g3/four,dd(1),dd(2),dd(3),del)
              index_p5(1)=0 
              cases=1
              If(cos_ini.ne.mu_tp1)then  !kt!=0 
                  tinf=b0/(four*(cos_ini-mu_tp1))+b1/four
                  call weierstrass_int_J3(tinf,infinity,dd,del,a4,b4,index_p5,rff_p,integ5,cases)
                  PI0=integ5(1) 
              else
                  PI0=zero !kt=0  
              endif
              If(kt.lt.zero)then
                  PI0=-PI0  
              endif
              mucos=mu_tp1+b0/(four*weierstrassP(p+PI0,g2,g3,dd,del)-b1)
              !write(*,*)'mu=',cos_ini,mu_tp1,mu_tp2,weierstrassP(p+PI0,g2,g3,dd,del),
              !tinf!tinf,infinity,g2,g3,a4,b4,p4,fzero
              !If cos_ini = 0,q = 0,and mu_tp1 = 0,so b0 = 0,
              ! so mucos eq mu_tp1 eq 0.
              count_num=count_num+1
          endif
          !***************************************************************************************************** 
      else
          If(kp.eq.kp_1.and.kt.eq.kt_1.and.lambda.eq.lambda_1.and.q.eq.q_1.and. mve.eq.mve_1 .and. &
                   sin_ini.eq.sin_ini_1.and.cos_ini.eq.cos_ini_1.and.a_spin.eq.a_spin_1)then
          !***************************************************************************************************** 
              If(kp.eq.zero .and. kt.eq.zero .and. abs(cos_ini).eq.one)then
                  IF(mve .LE. one)THEN    !In this case we have lambda=0, q=a^2(m^2-1)                
                      mucos=cos_ini         !so \Theta=(1-mu^2)(q-mu^2a^2(m^2-1))=a^2(m^2-1)(1-mu^2)^2
                      return              !so if m>1 mu E (-1,1), if m<1 \Theta<0, so mu==+1 or -1.         
                  ENDIF                   !if m=1,then \Theta_ini=0, so \theta_dot=0, so mu remains to
              endif                       !be +1 or -1.   
              If(a_spin.EQ.zero .OR. mve .EQ. one)then
                  if(q.gt.zero)then 
                      If(kt.gt.zero)then
                          mucos=sin(asin(cos_ini*AA)-p*BB*AA)/AA 
                      else
                          If(kt.lt.zero)then
                              mucos=sin(asin(cos_ini*AA)+p*AA*BB)/AA
                          else
                              mucos=cos(p*AA*BB)*cos_ini 
                          endif 
                      endif
                  else
                      mucos=cos_ini
                  endif
              else
                  If(cos_ini.eq.zero.and.q.eq.zero)then
                      mucos=zero
                      return 
                  endif 
                  mucos=mu_tp1+b0/(four*weierstrassP(p+PI0,g2,g3,dd,del)-b1) 
              endif
              !************************************************************************* 
          else
              count_num=1 
              goto 10 
          endif 
      endif
      return
      end Function mucos


!********************************************************************************************
      Function radius(p,kvecr,lambda,q,mve,ep,a_spin,e,r_ini)
!============================================================================================
!*     PURPOSE:  Computes function r(p) defined in Table 2, 3 in Yang & Wang (2013), A&A.  
!*     INPUTS:   p--------------independent variable, which must be nonnegative.
!*               kvecr----------k_{r}, the inital r components of four momentum of a 
!*                              particle measured under an LNRF. 
!*               lambda,q,mve,ep-------constants of motion, defined by lambda=L_z/E, q=Q/E^2,
!*                                     mve = \mu_m/E, ep=epsilon/varepsilon. 
!*               a_spin,e--------------the spin and the electric charge of the black hole, 
!*                                     and -1 =< a_spin^2+e^2 <= 1 must be satisfied. 
!*               r_ini-----------------the initial radial coordinate of the particle.       
!*     OUTPUTS:  radius---------radial coordinate of particle corresponding to a given p. 
!*     ROUTINES CALLED: weierstrass_int_J3, mutp, root3.
!*     ACCURACY:   Machine.    
!*     AUTHOR:     Yang & Wang (2012)  
!*     DATE WRITTEN:  4 Jan 2012
!*     REVISIONS: ****************************************** 
      implicit none
      Double precision radius,p,a_spin,rhorizon,q,lambda,zero,integ5(5),&
          bc,cc,dc,ec,b0,b1,b2,b3,g2,g3,tobs,tinf,PI0,PI1,r_ini,cr,dr,integ05(5),&
          u,v,w,L1,L2,thorizon,m2,pinf,sn,cn,dn,a4,b4,one,two,four,PI2,ttp,sqt3,&
          integ15(5),three,six,nine,r_tp1,r_tp2,kvecr,tp2,tp,t_inf,PI0_total,s,&
          PI0_inf_obs,PI0_obs_hori,PI01,PI0_total_2,rff_p,mve,mve_1,come,rinf,tp1,&
          ac1,bc1,cc1,alpha1,alpha2,c_temp,e,ep,ar(5),e_1,ep_1
      Double precision kvecr_1,lambda_1,q_1,p_1,a_spin_1,r_ini_1
      parameter(zero=0.D0,one=1.D0,two=2.D0,four=4.D0,three=3.D0,six=6.D0,nine=9.D0)
      complex*16 bb(1:4),dd(3)
      integer ::  reals,i,p4,cases_int,del,index_p5(5),cases,count_num=1
      logical :: r_ini_eq_rtp,indrhorizon,r1eqr2
      save  kvecr_1,lambda_1,q_1,a_spin_1,r_ini_1,r_tp1,r_tp2,reals,&
            r_ini_eq_rtp,indrhorizon,cases,bb,rhorizon,b0,b1,b2,b3,g2,g3,dd,del,cc,tobs,tp2,&
            thorizon,tinf,PI0,u,w,v,L1,L2,m2,t_inf,pinf,a4,b4,PI0_total,PI0_inf_obs,PI0_obs_hori,&
            PI0_total_2,mve_1,come,r1eqr2,e_1,ep_1
 
      IF(p.lt.zero)THEN
          Write(*,*)'radius(): p you given is <0, which is not valid and the code should be stopped.'
          Write(*,*)'Please input a positive one and try it again.'
          STOP
      ENDIF   
20    continue
      If(count_num.eq.1)then
          kvecr_1=kvecr 
          lambda_1=lambda
          q_1=q
          mve_1=mve
          ep_1=ep 
          a_spin_1=a_spin
          e_1=e
          r_ini_1=r_ini
          !********************************************************************************************* 
          rhorizon=one+sqrt(one-a_spin*a_spin-e*e)
          a4=zero
          b4=one 
          r_ini_eq_rtp=.false.
          indrhorizon=.false.
          IF(mve .EQ. one)THEN
               radius=radius_mb(p,kvecr,lambda,q,mve,ep,a_spin,e,r_ini) 
               RETURN                 
          ENDIF
          call radiustp(kvecr,a_spin,e,r_ini,lambda,q,mve,ep,r_tp1,r_tp2,&
                                     reals,r_ini_eq_rtp,indrhorizon,r1eqr2,cases,bb)
          IF(r1eqr2)THEN
              radius = r_ini
              return
          ENDIF
          come = mve*mve-one
          IF(reals .eq. 0)write(*,*)'ssssssssssss='
          !write(*,*)'kkkkkk=',reals,cases,come,mve,r_tp1,r_tp2,r_ini
          If(reals.ne.0)then  
              ar(1)=-come 
              ar(2)=two*(mve*mve+e*ep) 
              ar(3)=-(a_spin*a_spin*come+lambda*lambda+q+e*e*(mve*mve-ep*ep)) 
              ar(4)=two*(q+(lambda-a_spin)*(lambda-a_spin)+a_spin*e*ep*(a_spin-lambda)) 
              ar(5)=-q*(a_spin*a_spin+e*e)-e*e*(a_spin-lambda)**two 

              b0 = four*r_tp1**three*ar(1)+three*ar(2)*r_tp1**two+two*ar(3)*r_tp1+ar(4)
              b1 = two*r_tp1**two*ar(1)+ar(2)*r_tp1+ar(3)/three
              b2 = four/three*r_tp1*ar(1)+ar(2)/three
              b3 = ar(1)
              g2 = three/four*(b1*b1-b0*b2)
              g3 = (three*b0*b1*b2-two*b1*b1*b1-b0*b0*b3)/16.D0
 
              rinf = r_tp1  
              tp1 = infinity
              If(r_ini-rinf.ne.zero)then 
                  tobs=b0/four/(r_ini-rinf)+b1/four
              else
                  tobs=infinity
              endif 
              If(rhorizon-rinf.ne.zero)then
                  thorizon=b1/four+b0/four/(rhorizon-rinf)
              else
                  thorizon=infinity  
              endif 
              tp2=b0/four/(r_tp2-rinf)+b1/four 
              tinf=b1/four    

              call root3(zero,-g2/four,-g3/four,dd(1),dd(2),dd(3),del) 
              index_p5(1)=0 
              cases_int=1
              call weierstrass_int_J3(tobs,tp1,dd,del,a4,b4,index_p5,rff_p,integ05,cases_int) 
              PI0=integ05(1)
              select case(cases)
              case(1)
                  IF(kvecr.ge.zero)THEN
                      call weierstrass_int_J3(tinf,tobs,dd,del,a4,b4,index_p5,rff_p,integ05,cases_int)
                      PI0_inf_obs=integ05(1)
                      If(p.lt.PI0_inf_obs)then 
                          radius=r_tp1+b0/(four*weierstrassP(p+PI0,g2,g3,dd,del)-b1)  
                      else
                          radius=infinity !Goto infinity, far away.
                      endif
                  ELSE
                      IF(.NOT. indrhorizon)THEN
                          call weierstrass_int_J3(tinf,tp1,dd,del,a4,b4,index_p5,rff_p,integ15,cases_int) 
                          PI0_total=PI0+integ15(1)
                          If(p.lt.PI0_total)then  
                              radius=r_tp1+b0/(four*weierstrassP(p-PI0,g2,g3,dd,del)-b1) 
                          else
                              radius=infinity  !Goto infinity, far away.
                          endif                    
                      ELSE
                          call weierstrass_int_J3(tobs,thorizon,dd,del,a4,b4,index_p5,rff_p,integ05,cases_int) 
                          PI0_obs_hori=integ05(1) 
                          If(p.lt.PI0_obs_hori)then 
                              radius=r_tp1+b0/(four*weierstrassP(p-PI0,g2,g3,dd,del)-b1)  
                          else
                              radius=rhorizon !Fall into black hole.
                          endif 
                      ENDIF
                  ENDIF 
              case(2)
                  If(.not.indrhorizon)then
                      If(kvecr.lt.zero)then
                          PI01=-PI0
                      else
                          PI01=PI0 
                      endif 
                      radius=r_tp1+b0/(four*weierstrassP(p+PI01,g2,g3,dd,del)-b1)  
                  else 
                      If(kvecr.le.zero)then
                          call weierstrass_int_J3(tobs,thorizon,dd,del,a4,b4,index_p5,rff_p,integ15,cases_int) 
                          PI0_obs_hori = integ15(1)  
                          If(p.lt.PI0_obs_hori)then  
                              radius=r_tp1+b0/(four*weierstrassP(p-PI0,g2,g3,dd,del)-b1)  
                          else
                              radius=rhorizon !Fall into black hole.
                          endif 
                      else
                          call weierstrass_int_J3(tp2,thorizon,dd,del,a4,b4,index_p5,rff_p,integ15,cases_int) 
                          call weierstrass_int_J3(tp2,tobs,dd,del,a4,b4,index_p5,rff_p,integ5,cases_int)
                          PI0_total_2=integ15(1)+integ5(1)
                          If(p.lt.PI0_total_2)then 
                              radius=r_tp1+b0/(four*weierstrassP(p+PI0,g2,g3,dd,del)-b1) 
                          else
                              radius=rhorizon !Fall into black hole.
                          endif 
                      endif 
                   endif
              end select 
          else
              IF(-come .GT. zero)THEN
                  u=real(bb(4))
                  w=abs(aimag(bb(4)))
                  v=real(bb(3))
                  s=abs(aimag(bb(3)))
                  ac1=s*s
                  bc1=w*w+s*s+(u-v)*(u-v)
                  cc1=w*w 
                  L1=(bc1+dsqrt(bc1*bc1-four*ac1*cc1))/two/ac1            
                  L2=(bc1-dsqrt(bc1*bc1-four*ac1*cc1))/two/ac1 
                  alpha1=(L1*v-u)/(L1-one)
                  alpha2=(L2*v-u)/(L2-one)
                  thorizon = dsqrt((L1-one)/(L1-L2))*(rhorizon-alpha1)/dsqrt((rhorizon-v)**2+ac1)  
                  tobs = dsqrt((L1-one)/(L1-L2))*(r_ini-alpha1)/dsqrt((r_ini-v)**2+ac1)  
                  t_inf = dsqrt((L1-one)/(L1-L2))
                  m2 = (L1-L2)/L1   
                  pinf = EllipticF(tobs,m2)
                  IF(kvecr .LT. zero)THEN 
                      pinf=-pinf
                  ENDIF    
                  call sncndn(p*s*sqrt(-come*L1)+pinf,one-m2,sn,cn,dn)
                  If(kvecr.lt.zero)then
                      PI0=( abs(pinf)-EllipticF(thorizon,m2) )/s/dsqrt(-come*L1) 
                      if(p.lt.PI0)then
                          radius=v+(-u+v+s*(L1-L2)*sn*abs(cn))/((L1-L2)*sn**two-(L1-one))
                      else
                          radius=rhorizon
                      endif      
                  else
                      PI0=( EllipticF(t_inf,m2)-abs(pinf) )/s/dsqrt(-come*L1)
                      if(p.lt.PI0)then
                          radius=u+(-u+v-s*(L1-L2)*sn*abs(cn))/((L1-L2)*sn**two-(L1-one)) 
                      else
                          radius=infinity
                      endif
                  endif
              ENDIF

              IF(-come .LT. zero)THEN
                  u=real(bb(4))
                  w=abs(aimag(bb(4)))
                  v=real(bb(3))
                  s=abs(aimag(bb(3)))
                  ac1=s*s
                  bc1=-w*w-s*s-(u-v)*(u-v)
                  cc1=w*w 
                  L1=(bc1+dsqrt(bc1*bc1-four*ac1*cc1))/two/ac1            
                  L2=(bc1-dsqrt(bc1*bc1-four*ac1*cc1))/two/ac1 
                  alpha1=(L1*v+u)/(L1+one)
                  alpha2=(L2*v+u)/(L2+one)  
                  thorizon = dsqrt( (L1-L2)/(-L2*(one+L1)) )*(rhorizon-alpha1)/dsqrt( (rhorizon-u)**2+cc1 ) 
                  tobs = dsqrt( (L1-L2)/(-L2*(one+L1)) )*(r_ini-alpha1)/dsqrt( (r_ini-u)**2+cc1 )  
                  t_inf = dsqrt( (L1-L2)/(-L2*(one+L1)) ) 
                  m2 = (L2-L1)/L2
                  pinf = EllipticF(tobs,m2)
                  IF(kvecr .LT. zero)THEN 
                      pinf=-pinf
                  ENDIF    
                  call sncndn(p*s*dsqrt(-come*L2)+pinf,one-m2,sn,cn,dn)
                  c_temp = L2*(one+L1)/(L1-L2)
                  If(kvecr.lt.zero)then
                      PI0=( abs(pinf)-EllipticF(thorizon,m2) )/s/dsqrt(-come*L2) 
                      if(p.lt.PI0)then
                          radius=u+(-L1*(u-v)/(one+L1)+w*sn*dsqrt(one-(sn*c_temp)**2))/(one+sn*sn*c_temp)
                      else
                          radius=rhorizon
                      endif      
                  else
                      PI0=( EllipticF(t_inf,m2)-abs(pinf) )/s/dsqrt(-come*L2)
                      if(p.lt.PI0)then
                          radius=u+(-L1*(u-v)/(one+L1)-w*sn*dsqrt(one-(sn*c_temp)**2))/(one+sn*sn*c_temp) 
                      else
                          radius=infinity
                      endif
                  endif
              ENDIF       
          endif
          count_num=count_num+1
      else
          If(kvecr.eq.kvecr_1.and.lambda.eq.lambda_1.and.q.eq.q_1.and.mve_1.eq.mve.and.&
            a_spin.eq.a_spin_1.and.e.eq.e_1.and.r_ini.eq.r_ini_1.and.ep.eq.ep_1)then
      !***************************************************************************************************
            IF(r1eqr2)THEN
                radius = r_ini
                return
            ENDIF
            IF(mve .EQ. one)THEN
                radius=radius_mb(p,kvecr,lambda,q,mve,ep,a_spin,e,r_ini) 
                RETURN                  
            ENDIF 
            If(reals.ne.0)then    
              select case(cases)
              case(1)
                  IF(kvecr.ge.zero)THEN 
                      If(p.lt.PI0_inf_obs)then 
                          radius=r_tp1+b0/(four*weierstrassP(p+PI0,g2,g3,dd,del)-b1)  
                      else
                          radius=infinity !Goto infinity, far away.
                      endif
                  ELSE
                      IF(.NOT. indrhorizon)THEN 
                          If(p.lt.PI0_total)then     
                              radius=r_tp1+b0/(four*weierstrassP(p-PI0,g2,g3,dd,del)-b1)      
                          else
                              radius=infinity  !Goto infinity, far away.
                          endif                      
                      ELSE 
                          If(p.lt.PI0_obs_hori)then 
                              radius=r_tp1+b0/(four*weierstrassP(p-PI0,g2,g3,dd,del)-b1)   
                          else
                              radius=rhorizon !Fall into black hole.
                          endif 
                      ENDIF
                  ENDIF 
              case(2)
                  If(.not.indrhorizon)then
                      If(kvecr.lt.zero)then
                          PI01=-PI0
                      else
                          PI01=PI0 
                      endif 
                      radius=r_tp1+b0/(four*weierstrassP(p+PI01,g2,g3,dd,del)-b1)       
                  else 
                      If(kvecr.le.zero)then  
                          If(p.lt.PI0_obs_hori)then  
                              radius=r_tp1+b0/(four*weierstrassP(p-PI0,g2,g3,dd,del)-b1)   
                          else
                              radius=rhorizon !Fall into black hole.
                          endif   
                      else 
                          If(p.lt.PI0_total_2)then 
                              radius=r_tp1+b0/(four*weierstrassP(p+PI0,g2,g3,dd,del)-b1)   
                          else
                              radius=rhorizon !Fall into black hole.
                          endif 
                      endif         
                   endif
              end select    
            else
              IF(-come .GT. zero)THEN    
                  call sncndn(p*s*sqrt(-come*L1)+pinf,one-m2,sn,cn,dn)
                  If(kvecr.lt.zero)then  
                      if(p.lt.PI0)then
                          radius=v+(-u+v+s*(L1-L2)*sn*abs(cn))/((L1-L2)*sn**two-(L1-one))
                      else
                          radius=rhorizon
                      endif      
                  else 
                      if(p.lt.PI0)then
                          radius=u+(-u+v-s*(L1-L2)*sn*abs(cn))/((L1-L2)*sn**two-(L1-one)) 
                      else
                          radius=infinity
                      endif
                  endif
              ENDIF

              IF(-come .LT. zero)THEN 
                  call sncndn(p*s*dsqrt(-come*L2)+pinf,one-m2,sn,cn,dn)
                  c_temp = L2*(one+L1)/(L1-L2)
                  If(kvecr.lt.zero)then  
                      if(p.lt.PI0)then
                          radius=u+(-L1*(u-v)/(one+L1)+w*sn*dsqrt(one-(sn*c_temp)**2))/(one+sn*sn*c_temp)
                      else
                          radius=rhorizon
                      endif      
                  else 
                      if(p.lt.PI0)then
                          radius=u+(-L1*(u-v)/(one+L1)-w*sn*dsqrt(one-(sn*c_temp)**2))/(one+sn*sn*c_temp) 
                      else
                          radius=infinity
                      endif
                  endif
              ENDIF      
            endif
      !***************************************************************************************************
          else
              count_num=1 
              goto  20
          endif
      endif 
      return   
      End function radius


!********************************************************************************************
      FUNCTION radius_mb(p,kvecr,lambda,q,mve,ep,a_spin,e,r_ini)  
!============================================================================================
!*     PURPOSE:  Computes function r(p) of cases 4 and 5 defined in Table 2, 3 in Yang & Wang 
!*               (2013), A&A, in these two cases constant of motion mve = 1.  
!*     INPUTS:   p--------------independent variable, which must be nonnegative.
!*               kvecr----------k_{r}, the inital r components of four momentum of a 
!*                              particle measured under an LNRF. 
!*               lambda,q,mve,ep-------constants of motion, defined by lambda=L_z/E, q=Q/E^2,
!*                                     mve = \mu_m/E, ep=epsilon/varepsilon. 
!*               a_spin,e--------------the spin and the electric charge of the black hole, 
!*                                     and -1 =< a_spin^2+e^2 <= 1 must be satisfied. 
!*               r_ini-----------------the initial radial coordinate of the particle.       
!*     OUTPUTS:  radius----------------radial coordinate of particle corresponding to a given p
!*                                     with mve = 1. 
!*     ROUTINES CALLED: weierstrass_int_J3, mutp, root3.
!*     ACCURACY:   Machine.    
!*     AUTHOR:     Yang & Wang (2012)  
!*     DATE WRITTEN:  4 Jan 2012
!*     REVISIONS: ****************************************** 
      USE constants
      implicit none
      Double precision radius_mb,p,a_spin,rhorizon,q,lambda,integ5(5),&
          bc,cc,dc,ec,b0,b1,b2,b3,g2,g3,tobs,tinf,PI0,PI1,r_ini,cr,dr,integ05(5),&
          u,v,w,L1,L2,thorizon,m2,pinf,sn,cn,dn,a4,b4,PI2,ttp,sqt3,mve_1,&
          integ15(5),r_tp1,r_tp2,kvecr,tp2,tp,t_inf,PI0_total,s,mve,&
          PI0_inf_obs,PI0_obs_hori,PI01,PI0_total_2,rff_p,rinf,tp1,&
          ac1,bc1,cc1,alpha1,alpha2,c_temp,e,ep,e_1,ep_1,&
          xi,e1,e2,e3,xi_ini,xi_tp1,xi_tp2,xi_horizon,PI0_ini_hori,period
      Double precision kvecr_1,lambda_1,q_1,p_1,a_spin_1,r_ini_1 
      complex*16 bb(1:4),dd(3)
      integer ::  reals,i,p4,cases_int,del,index_p5(5),cases,count_num=1
      logical :: r_ini_eq_rtp,indrhorizon,r1eqr2
      save  kvecr_1,lambda_1,q_1,a_spin_1,r_ini_1,r_tp1,r_tp2,reals,&
            r_ini_eq_rtp,indrhorizon,cases,bb,rhorizon,b0,b1,b2,b3,g2,g3,dd,del,cc,tobs,tp2,&
            thorizon,tinf,PI0,u,w,v,L1,L2,m2,t_inf,pinf,a4,b4,PI0_total,PI0_inf_obs,PI0_obs_hori,&
            PI0_total_2,mve_1,r1eqr2,e_1,ep_1,e1,e2,e3,xi_ini,xi_tp1,&
            xi_tp2,xi_horizon,PI0_ini_hori,period

90    continue
      If(count_num.eq.1)then
          kvecr_1=kvecr 
          lambda_1=lambda
          q_1=q 
          ep_1=ep
          a_spin_1=a_spin
          e_1=e
          r_ini_1=r_ini
          !********************************************* 
          rhorizon=one+sqrt(one-a_spin*a_spin-e*e)
          a4=zero
          b4=one
          cc=a_spin**2-lambda**2-q
          r_ini_eq_rtp=.false.
          indrhorizon=.false.
          call radiustp(kvecr,a_spin,e,r_ini,lambda,q,mve,ep,r_tp1,r_tp2,&
                                     reals,r_ini_eq_rtp,indrhorizon,r1eqr2,cases,bb) 
          IF(r1eqr2)THEN
              radius_mb = r_ini
              return
          ENDIF
          IF(reals .eq. 0)write(*,*)'ssssssssssss='
          !write(*,*)'kkkkkk=',reals,cases,come,mve,r_tp1,r_tp2,r_ini    
          b0=two*(one+e*ep)  
          b1=-(lambda*lambda+q+e*e*(one-ep*ep))/three
          b2=two*(q+(lambda-a_spin)*(lambda-a_spin)+a_spin*e*ep*(a_spin-lambda))/three
          b3=-q*(a_spin*a_spin+e*e)-e*e*(a_spin-lambda)**two
          g2 = three/four*(b1*b1-b0*b2)
          g3 = (three*b0*b1*b2-two*b1*b1*b1-b0*b0*b3)/16.D0  
  
          tobs = b0/four*r_ini+b1/four
          thorizon = b0/four*rhorizon+b1/four  
          tinf =infinity   

          call root3(zero,-g2/four,-g3/four,dd(1),dd(2),dd(3),del) 
          e1 = real(dd(1))
          e2 = real(dd(2))
          e3 = real(dd(3))
          period = halfperiodwp(g2,g3,dd,del)
          index_p5(1)=0 
          cases_int=1
          call weierstrass_int_J3(tobs,tinf,dd,del,a4,b4,index_p5,rff_p,integ05,cases_int) 
          PI0=integ05(1) 
              select case(cases)
              case(1)
                  IF(kvecr.ge.zero)THEN
                      If(b0.ge.zero)then
                          !call weierstrass_int_J3(tobs,tinf,dd,del,a4,b4,index_p5,rff_p,integ05,cases_int)
                          PI0_inf_obs = PI0 !integ05(1)
                          If(p.lt.PI0_inf_obs)then 
                              radius_mb = (four*weierstrassP(p-sign(one,b0)*PI0,g2,g3,dd,del)-b1)/b0  
                          else
                              radius_mb = infinity !Goto infinity, far away.
                          endif
                      else
                          call weierstrass_int_J3(e1,tobs,dd,del,a4,b4,index_p5,rff_p,integ05,cases_int)
                          call weierstrass_int_J3(e1,thorizon,dd,del,a4,b4,index_p5,rff_p,integ15,cases_int)
                          PI0_ini_hori = integ05(1)+integ15(1)
                          If(p .lt. PI0_ini_hori)then
                              radius_mb = (four*weierstrassP(p-sign(one,b0)*PI0,g2,g3,dd,del)-b1)/b0   
                          else
                              radius_mb = rhorizon ! particle already fall into the event horizon.
                          endif 
                      endif
                  ELSE
                      IF(b0.ge.zero)THEN
                          IF(.NOT. indrhorizon)THEN 
                              PI0_total = two*period-PI0
                              If(p.lt.PI0_total)then 
                                  radius_mb = (four*weierstrassP(p+sign(one,b0)*PI0,g2,g3,dd,del)-b1)/b0      
                              else
                                  radius_mb = infinity  !Goto infinity, far away.
                              endif                      
                          ELSE
                              call weierstrass_int_J3(thorizon,tobs,dd,del,a4,b4,index_p5,rff_p,integ05,cases_int) 
                              PI0_ini_hori=integ05(1) 
                              If(p.lt.PI0_ini_hori)then 
                                  radius_mb = (four*weierstrassP(p+sign(one,b0)*PI0,g2,g3,dd,del)-b1)/b0  
                              else
                                  radius_mb = rhorizon !Fall into black hole.
                              endif 
                          ENDIF
                      ELSE
                          call weierstrass_int_J3(tobs,thorizon,dd,del,a4,b4,index_p5,rff_p,integ05,cases_int) 
                          PI0_ini_hori=integ05(1)    
                          If(p.lt.PI0_ini_hori)then 
                              radius_mb = (four*weierstrassP(p+sign(one,b0)*PI0,g2,g3,dd,del)-b1)/b0  
                          else
                              radius_mb = rhorizon !Fall into black hole.
                          endif  
                      ENDIF
                  ENDIF 
              case(2)
                  index_p5(1)=0 
                  cases_int=1
                  xi_ini = e2-(e1-e2)*(e2-e3)/(tobs-e2)
                  xi_tp1 = e1 !e2-(e1-e2)*(e2-e3)/(e3-e2) ! for tp1 = e3, tp2 = e2.
                  xi_tp2 = infinity ! e2-(e1-e2)*(e2-e3)/(tp2-e2), for tp2=e2.
                  xi_horizon = e2-(e1-e2)*(e2-e3)/(thorizon-e2)
                  call weierstrass_int_J3(xi_ini,tinf,dd,del,a4,b4,index_p5,rff_p,integ05,cases_int)  
                  PI0=integ05(1) 
                  If(kvecr.ge.zero)then
                      PI01 = -PI0*sign(one,b0)
                  else
                      PI01 = PI0*sign(one,b0) 
                  endif 
                  If(.not.indrhorizon)then
                      radius_mb = (four*e2-b1-four*(e1-e2)*(e2-e3)/(weierstrassP(p+PI01,g2,g3,dd,del)-e2))/b0      
                  else
                      If(b0.ge.zero)then
                          If(kvecr.le.zero)then
                              call weierstrass_int_J3(xi_horizon,xi_ini,dd,del,a4,b4,index_p5,rff_p,integ15,cases_int) 
                              PI0_ini_hori = integ15(1)  
                              If(p.lt.PI0_ini_hori)then 
                                  radius_mb = (four*e2-b1-four*(e1-e2)*(e2-e3)/(weierstrassP(p+PI01,g2,g3,dd,del)-e2))/b0   
                              else
                                  radius_mb = rhorizon !Fall into black hole.
                              endif   
                          else
                              call weierstrass_int_J3(xi_horizon,xi_tp2,dd,del,a4,b4,index_p5,rff_p,integ15,cases_int)   
                              call weierstrass_int_J3(xi_ini,xi_tp2,dd,del,a4,b4,index_p5,rff_p,integ5,cases_int) 
                              PI0_ini_hori = integ15(1)+integ5(1)
                              If(p.lt.PI0_ini_hori)then 
                                  radius_mb = (four*e2-b1-four*(e1-e2)*(e2-e3)/(weierstrassP(p+PI01,g2,g3,dd,del)-e2))/b0  
                              else
                                  radius_mb = rhorizon !Fall into black hole.
                              endif 
                          endif   
                      else
                          If(kvecr.le.zero)then
                              call weierstrass_int_J3(xi_ini,xi_horizon,dd,del,a4,b4,index_p5,rff_p,integ15,cases_int) 
                              PI0_ini_hori = integ15(1)  
                              If(p.lt.PI0_ini_hori)then 
                                  radius_mb = (four*e2-b1-four*(e1-e2)*(e2-e3)/(weierstrassP(p+PI01,g2,g3,dd,del)-e2))/b0   
                              else
                                  radius_mb = rhorizon !Fall into black hole.
                              endif   
                          else
                              call weierstrass_int_J3(e1,xi_horizon,dd,del,a4,b4,index_p5,rff_p,integ15,cases_int)   
                              call weierstrass_int_J3(e1,xi_ini,dd,del,a4,b4,index_p5,rff_p,integ5,cases_int) 
                              PI0_ini_hori = integ15(1)+integ5(1)
                              If(p.lt.PI0_ini_hori)then 
                                  radius_mb = (four*e2-b1-four*(e1-e2)*(e2-e3)/(weierstrassP(p+PI01,g2,g3,dd,del)-e2))/b0  
                              else
                                  radius_mb = rhorizon !Fall into black hole.
                              endif 
                          endif   
                      endif         
                  endif 
              end select  
          count_num = count_num+1
      else
          If(kvecr.eq.kvecr_1.and.lambda.eq.lambda_1.and.q.eq.q_1.and.mve_1.eq.mve.and.&
          a_spin.eq.a_spin_1.and.ep_1.eq.ep.and.r_ini.eq.r_ini_1.and.e_1.eq.e)then
      !***************************************************************************** 
              IF(r1eqr2)THEN
                  radius_mb = r_ini
                  return
              ENDIF      
              select case(cases)
              case(1)
                  IF(kvecr.ge.zero)THEN
                      If(b0.ge.zero)then 
                          If(p.lt.PI0_inf_obs)then 
                              radius_mb = (four*weierstrassP(p-sign(one,b0)*PI0,g2,g3,dd,del)-b1)/b0  
                          else
                              radius_mb = infinity !Goto infinity, far away.
                          endif
                      else 
                          If(p .lt. PI0_ini_hori)then
                              radius_mb = (four*weierstrassP(p-sign(one,b0)*PI0,g2,g3,dd,del)-b1)/b0   
                          else
                              radius_mb = rhorizon ! particle already fall into the event horizon.
                          endif 
                      endif
                  ELSE
                      IF(b0.ge.zero)THEN
                          IF(.NOT. indrhorizon)THEN  
                              If(p.lt.PI0_total)then 
                                  radius_mb = (four*weierstrassP(p+sign(one,b0)*PI0,g2,g3,dd,del)-b1)/b0      
                              else
                                  radius_mb = infinity  !Goto infinity, far away.
                              endif                      
                          ELSE 
                              If(p.lt.PI0_ini_hori)then 
                                  radius_mb = (four*weierstrassP(p+sign(one,b0)*PI0,g2,g3,dd,del)-b1)/b0  
                              else
                                  radius_mb = rhorizon !Fall into black hole.
                              endif 
                          ENDIF
                      ELSE    
                          If(p.lt.PI0_ini_hori)then 
                              radius_mb = (four*weierstrassP(p+sign(one,b0)*PI0,g2,g3,dd,del)-b1)/b0  
                          else
                              radius_mb = rhorizon !Fall into black hole.
                          endif  
                      ENDIF
                  ENDIF 
              case(2) 
                  If(.not.indrhorizon)then
                      radius_mb = (four*e2-b1-four*(e1-e2)*(e2-e3)/(weierstrassP(p+PI01,g2,g3,dd,del)-e2))/b0      
                  else
                      If(b0.ge.zero)then
                          If(kvecr.le.zero)then  
                              If(p.lt.PI0_ini_hori)then 
                                  radius_mb = (four*e2-b1-four*(e1-e2)*(e2-e3)/(weierstrassP(p+PI01,g2,g3,dd,del)-e2))/b0   
                              else
                                  radius_mb = rhorizon !Fall into black hole.
                              endif   
                          else 
                              If(p.lt.PI0_ini_hori)then 
                                  radius_mb = (four*e2-b1-four*(e1-e2)*(e2-e3)/(weierstrassP(p+PI01,g2,g3,dd,del)-e2))/b0  
                              else
                                  radius_mb = rhorizon !Fall into black hole.
                              endif 
                          endif   
                      else
                          If(kvecr.le.zero)then  
                              If(p.lt.PI0_ini_hori)then 
                                  radius_mb = (four*e2-b1-four*(e1-e2)*(e2-e3)/(weierstrassP(p+PI01,g2,g3,dd,del)-e2))/b0   
                              else
                                  radius_mb = rhorizon !Fall into black hole.
                              endif   
                          else 
                              If(p.lt.PI0_ini_hori)then 
                                  radius_mb = (four*e2-b1-four*(e1-e2)*(e2-e3)/(weierstrassP(p+PI01,g2,g3,dd,del)-e2))/b0  
                              else
                                  radius_mb = rhorizon !Fall into black hole.
                              endif 
                          endif   
                      endif         
                  endif 
              end select  
      !********************************************************************* 
          else
              count_num=1 
              goto  90
          endif
      endif 
      return   
      END FUNCTION radius_mb  


!********************************************************************************************************
      SUBROUTINE INTTPART(p,kp,kt,lambda,q,mve,sin_ini,cos_ini,a_spin,phyt,timet,mucos,t1,t2)    
!********************************************************************************************************
!*     PURPOSE:  Computes \mu part of integrals in coordinates \phi, t and proper time \sigma,
!*               expressed in Table 5 and 6 in Yang & Wang (2013), A&A.    
!*     INPUTS:   p--------------independent variable, which must be nonnegative.
!*               kp-------------k_{\phi}, initial \phi component of four momentum of a particle
!*                              measured under an LNRF.
!*               kt-------------k_{\theta}, initial \theta component of four momentum of a particle
!*                              measured under an LNRF. 
!*               lambda,q,mve-------constants of motion, defined by lambda=L_z/E, q=Q/E^2, mve = \mu_m/E.
!*               sin_ini,cos_ini-------sin_ini=sin(\theta_{ini}), cos_ini=cos(\theta_{ini}), where 
!*                                     \theta_{ini} is the initial \theta coordinate of the particle.
!*               a_spin---------spin of black hole, on interval [-1,1].        
!*     OUTPUTS:  phyt-----------value of integral \phi_\theta.
!*               timet----------value of integral \t_\theta. And \sigma_\theta=time_\theta.
!*               mucos----------value of function \mu(p).
!*               t1,t2----------number of time_0 of photon meets turning points \mu_tp1 and \mu_tp2
!*                              respectively for a given p.            
!*     ROUTINES CALLED: mutp, root3, weierstrass_int_J3 
!*     ACCURACY:   Machine.    
!*     AUTHOR:     Yang & Wang (2012)  
!*     DATE WRITTEN:  5 Jan 2012
!*     REVISIONS: ***************************************************************************************
      USE constants
      IMPLICIT NONE
      Double precision phyt,timet,kp,kt,p,sin_ini,cos_ini,a_spin,lambda,q,mu_tp1,tposition,tp2,mu,tmu,p1J2,&
             bc,cc,dc,ec,b0,b1,b2,b3,g2,g3,tobs,p1,p2,pp,c_add,c_m,a_add,a_m,p1J1,come,&
             p1I0,a4,b4 ,delta,mu_tp2,mutemp ,integ5(5),integ(5),rff_p,tp1,&
             integ15(5),pp2,f1234(4),PI0,integ05(5),fzero,mu2p,PI01,h,p1_t,p2_t,pp_t,p1_phi,&
             p2_phi,pp_phi,radius,mtp1,mtp2,mucos,sqt3,difference,p_mt1_mt2,&
             PI1_phi,PI2_phi,PI1_time,PI2_time,PI2_p,mve,PI1_p,p1_temp,p2_temp
      Double precision kp_1,kt_1,lambda_1,q_1,sin_ini_1,cos_ini_1,a_spin_1,mve_1
      !parameter(zero=0.D0,two=2.D0,four=4.D0,one=1.D0,three=3.D0)
      integer ::  t1,t2,i,j,reals,cases,p4,index_p5(5),del,cases_int,N_temp,count_num=1
      complex*16 bb(1:4),dd(3)
      logical :: err,mobseqmtp
      save  kp_1,kt_1,lambda_1,q_1,sin_ini_1,cos_ini_1,a_spin_1,a4,b4,mu_tp1,mu_tp2,reals,&
            mobseqmtp,b0,b1,b2,b3,g2,g3,dd,del,PI0,c_m,c_add,a_m,a_add,tp2,tobs,h,p_mt1_mt2,&
            PI1_phi,PI2_phi,PI1_time,PI2_time,PI2_p,mve_1,come,tp1,PI1_p

      IF(p.lt.zero)THEN
          Write(*,*)'INTTPART(): p you given is <0, which is not valid and the code should be stopped.'
          Write(*,*)'Please input a positive one and try it again.'
          STOP
      ENDIF
30    continue
      IF(count_num.eq.1)then 
          kp_1=kp
          kt_1=kt
          lambda_1=lambda
          q_1=q
          mve_1=mve 
          cos_ini_1=cos_ini
          sin_ini_1=sin_ini
          a_spin_1=a_spin
          t1=0
          t2=0  
     !***********************************************************************
          If(kp.eq.zero .and. kt.eq.zero .and. abs(cos_ini).eq.one)then
              IF(mve .LE. one)THEN    !In this case we have lambda=0, q=a^2(m^2-1)                
                  mucos=cos_ini         !so \Theta=(1-mu^2)(q-mu^2a^2(m^2-1))=a^2(m^2-1)(1-mu^2)^2
                  timet=zero          !so if m>1 mu E (-1,1), if m<1 \Theta<0, so mu==+1 or -1. 
                  phyt=zero           !if m=1,then \Theta_ini=0, so \theta_dot=0, so mu remains to
                  count_num=count_num+1
                  return              !be +1 or -1.                                    
              ENDIF                   
          endif       
          If(cos_ini.eq.zero .and. abs(lambda).lt.abs(a_spin*dsqrt( abs(mve*mve-one) )) .and. q.eq.zero &
             .and.abs(mve).lt.one)then
              timet=zero  !in this case, \Theta_mu=-mu^2a^2(m^2-1)(mu^2+/-mu0^2),mu0= thus mu=0 is     
              phyt=zero   !double root of equation of \Theta_mu=0, and the integrals about \theta
              mucos=zero  !thus diverge, which preveats the particle approaching to or eacapting  
              count_num=count_num+1  
              return      !from the equatorial plane.   
          endif
          mobseqmtp=.false.
          call mutp(kp,kt,sin_ini,cos_ini,a_spin,lambda,q,mve,mu_tp1,mu_tp2,reals,mobseqmtp) 
          If(mu_tp1.eq.zero)then
              !photons are confined in the equatorial plane, so the integrations about \theta are valished.
              timet=zero
              phyt=zero
              mucos=zero
              count_num=count_num+1  
              return
          endif
          !**************************************************************************
          If(a_spin.eq.zero .or. mve .eq.one)then 
              CALL phyt_schwatz(kp,kt,lambda,q,mve,p,sin_ini,cos_ini,a_spin,phyt,timet,mucos,t1,t2)
              count_num=count_num+1
              return
          endif 
          a4=zero
          b4=one

          come = mve*mve-one  
          b0=four*a_spin**2*mu_tp1**3*come-two*mu_tp1*(a_spin**2*come+lambda**2+q)
          b1=two*a_spin**2*mu_tp1**2*come-(a_spin**2*come+lambda**2+q)/three
          b2=four/three*a_spin**2*mu_tp1*come
          b3=a_spin**2*come
          g2=three/four*(b1**2-b0*b2)
          g3=(three*b0*b1*b2-two*b1**3-b0**2*b3)/16.D0  
          call root3(zero,-g2/four,-g3/four,dd(1),dd(2),dd(3),del)

          If(cos_ini.ne.mu_tp1)then 
              tobs=b0/four/(cos_ini-mu_tp1)+b1/four
          else
              tobs=infinity
          endif
          tp1=infinity
          tp2=b0/four/(mu_tp2-mu_tp1)+b1/four
          If(mu_tp1-one.ne.zero)then
               c_m=b0/(four*(-one-mu_tp1)**2)
             c_add=b0/(four*(one-mu_tp1)**2) 
               a_m=b0/four/(-one-mu_tp1)+b1/four
             a_add=b0/four/(one-mu_tp1)+b1/four
          endif
          index_p5(1)=0
          cases_int=1
          call weierstrass_int_J3(tobs,tp1,dd,del,a4,b4,index_p5,rff_p,integ05,cases_int)
          PI0=integ05(1) 
          If(kt.lt.zero)then
              PI01=-PI0 
          else
              PI01=PI0
          endif
          tmu=weierstrassP(p+PI01,g2,g3,dd,del)
          mucos = mu_tp1+b0/(four*tmu-b1)
          h=-b1/four 
          !to get number of turn points of t1 and t2.
          !111111111*****************************************************************************************
          !mu=mu_tp+b0/(four*tmu-b1)
          call weierstrass_int_J3(tp2,tp1,dd,del,a4,b4,index_p5,rff_p,integ15,cases_int)
          call weierstrass_int_J3(tobs,tmu,dd,del,a4,b4,index_p5,rff_p,integ5,cases_int)  
          p_mt1_mt2=integ15(1)
          PI1_p=PI0
          PI2_p=p_mt1_mt2-PI0
          pp=integ5(1)
          p1=PI0-pp
          p2=p_mt1_mt2-p1
          PI1_phi=zero
          PI2_phi=zero
          PI1_time=zero
          PI2_time=zero 
!========================================================================
    !======== To determine the number of time_0 N_t1, N_t2 that the particle meets the
    !======== two turn points r_tp1, r_tp2 
                      If(mobseqmtp)then
                          p1_temp = zero
                          p2_temp = p_mt1_mt2
                          t1 = 0
                          t2 = 0
                          N_temp = 0
                          If(cos_ini.eq.mu_tp1)then
                              Do While(.TRUE.)
                                  IF(p1_temp.le.p .and. p.le.p2_temp)then
                                      EXIT
                                  ELSE
                                      N_temp = N_temp + 1
                                      p1_temp = p2_temp
                                      p2_temp = p2_temp + p_mt1_mt2
                                      t1 = t2
                                      t2 = N_temp-t1
                                  ENDIF
                              ENDDO
                          Else If(cos_ini.eq.mu_tp2)then
                              Do While(.TRUE.)
                                  IF(p1_temp.le.p .and. p.le.p2_temp)then
                                      EXIT
                                  ELSE
                                      N_temp = N_temp + 1
                                      p1_temp = p2_temp
                                      p2_temp = p2_temp + p_mt1_mt2
                                      t2 = t1
                                      t1 = N_temp-t2
                                  ENDIF
                              ENDDO
                          Endif 
                      Else
                          p1_temp = zero
                          t1 = 0
                          t2 = 0
                          N_temp = 0
                          If(kt.gt.zero)then
                              p2_temp = PI2_p
                              Do While(.TRUE.)
                                  IF(p1_temp.le.p .and. p.le.p2_temp)then
                                      EXIT
                                  ELSE
                                      N_temp = N_temp + 1
                                      p1_temp = p2_temp
                                      p2_temp = p2_temp + p_mt1_mt2
                                      t1 = t2
                                      t2 = N_temp-t1
                                  ENDIF
                              ENDDO
                          ENDIF
                          If(kt.lt.zero)then
                              p2_temp = PI1_p
                              Do While(.TRUE.)
                                  IF(p1_temp.le.p .and. p.le.p2_temp)then
                                      EXIT
                                  ELSE
                                      N_temp = N_temp + 1
                                      p1_temp = p2_temp
                                      p2_temp = p2_temp + p_mt1_mt2
                                      t2 = t1
                                      t1 = N_temp-t2
                                  ENDIF
                              ENDDO
                          ENDIF
                      Endif   
!========================================================================
          index_p5(1)=-1
          index_p5(2)=-2
          index_p5(3)=0
          index_p5(4)=-4
          index_p5(5)=0
          !*****pp part***************************************
          If(lambda.ne.zero)then 
              cases_int=2
              call weierstrass_int_J3(tobs,tmu,dd,del,-a_add,b4,index_p5,abs(pp),integ5,cases_int)
              call weierstrass_int_J3(tobs,tmu,dd,del,-a_m,b4,index_p5,abs(pp),integ15,cases_int)
              pp_phi=(pp/(one-mu_tp1**two)+(integ5(2)*c_add-integ15(2)*c_m)/two)*lambda
          else 
              pp_phi=zero   
          endif
          cases_int=4
          call weierstrass_int_J3(tobs,tmu,dd,del,h,b4,index_p5,abs(pp),integ,cases_int)
          pp_t=(pp*mu_tp1**two+integ(2)*mu_tp1*b0/two+integ(4)*b0**two/sixteen)*a_spin*a_spin
          !*****p1 part***************************************
          If(t1.eq.0)then 
              p1_phi=zero
              p1_t=zero
          else  
              If(lambda.ne.zero)then  
                  IF(PI1_phi .EQ. zero)THEN
                      cases_int=2 
                      call weierstrass_int_J3(tobs,infinity,dd,del,-a_add,b4,index_p5,PI0,integ5,cases_int)
                      call weierstrass_int_J3(tobs,infinity,dd,del,-a_m,b4,index_p5,PI0,integ15,cases_int)
                      PI1_phi=(PI0/(one-mu_tp1**two)+(integ5(2)*c_add-integ15(2)*c_m)/two)*lambda
                  ENDIF 
                  p1_phi=PI1_phi-pp_phi 
              else 
                  p1_phi=zero      
              endif 
              IF(PI1_time .EQ. zero)THEN  
                  cases_int=4  
                  call weierstrass_int_J3(tobs,infinity,dd,del,h,b4,index_p5,PI0,integ,cases_int)
                  PI1_time=(PI0*mu_tp1**two+integ(2)*mu_tp1*b0/two+integ(4)*b0**two/sixteen)*a_spin**two 
              ENDIF
              p1_t=PI1_time-pp_t 
          endif 
          !*****p2 part***************************************
          If(t2.eq.0)then
              p2_phi=zero
              p2_t=zero
          else
              IF(lambda.ne.zero)then  
                  IF(PI2_phi .EQ. zero)THEN  
                      cases_int=2 
                      call weierstrass_int_J3(tp2,tobs,dd,del,-a_add,b4,index_p5,PI2_p,integ5,cases_int)
                      call weierstrass_int_J3(tp2,tobs,dd,del,-a_m,b4,index_p5,PI2_p,integ15,cases_int)
                      PI2_phi=(PI2_p/(one-mu_tp1*mu_tp1)+(integ5(2)*c_add-integ15(2)*c_m)/two)*lambda 
                  ENDIF
                  p2_phi=PI2_phi+pp_phi
              ELSE
                  p2_phi=zero                
              ENDIF 
                
              IF(PI2_time .EQ. zero)THEN  
                  cases_int=4  
                  call weierstrass_int_J3(tp2,tobs,dd,del,h,b4,index_p5,PI2_p,integ,cases_int)
                  PI2_time=(PI2_p*mu_tp1**two+integ(2)*mu_tp1*b0/two+integ(4)*b0**two/sixteen)*a_spin**two 
              ENDIF 
              p2_t=PI2_time+pp_t   
          endif   
          !write(*,*)'kkkk=',pp_phi,p1_phi,p2_phi,t1,t2
 !**************************************************************
          If(mobseqmtp)then 
              If(cos_ini.eq.mu_tp1)then
                  phyt=-pp_phi+two*(t1*p1_phi+t2*p2_phi)
                  timet=-pp_t+two*(t1*p1_t+t2*p2_t) 
              else
                  phyt=pp_phi+two*(t1*p1_phi+t2*p2_phi)
                  timet=pp_t+two*(t1*p1_t+t2*p2_t)  
              endif
          else
              If(kt.lt.zero)then
                  phyt=pp_phi+two*(t1*p1_phi+t2*p2_phi) 
                  timet=pp_t+two*(t1*p1_t+t2*p2_t)
              endif
              If(kt.gt.zero)then  
                  phyt=-pp_phi+two*(t1*p1_phi+t2*p2_phi)
                  timet=-pp_t+two*(t1*p1_t+t2*p2_t) 
              endif
          endif
          count_num=count_num+1
      ELSE 
          If(kp_1.eq.kp.and.kt_1.eq.kt.and.lambda_1.eq.lambda.and.q_1.eq.q.and.sin_ini_1.eq.sin_ini&
          .and.cos_ini_1.eq.cos_ini.and.a_spin_1.eq.a_spin.and.mve_1.eq.mve)then   
 !***************************************************************************
          t1=0
          t2=0  
        !*********************************************************************** 
          If(kp.eq.zero .and. kt.eq.zero .and. abs(cos_ini).eq.one)then
              IF(mve .LE. one)THEN    !In this case we have lambda=0, q=a^2(m^2-1)                
                  mucos=cos_ini         !so \Theta=(1-mu^2)(q-mu^2a^2(m^2-1))=a^2(m^2-1)(1-mu^2)^2
                  timet=zero          !so if m>1 mu E (-1,1), if m<1 \Theta<0, so mu==+1 or -1. 
                  phyt=zero           !if m=1,then \Theta_ini=0, so \theta_dot=0, so mu remains to
                  return              !be +1 or -1.                                    
              ENDIF                   
          endif      
          If(cos_ini.eq.zero .and. abs(lambda).lt.abs(a_spin*dsqrt( abs(mve*mve-one) )) .and. &
              mve.lt.one .and. q.eq.zero)then
              timet=zero  !in this case, \Theta_mu=-mu^2a^2(m^2-1)(mu^2+/-mu0^2), thus mu=0 is     
              phyt=zero   !double root of equation of \Theta_mu=0, and the integrals about \theta
              mucos=zero  !thus diverge, which preveats the particle approaching to or eacapting    
              return      !from the equatorial plane.   
          endif  
          If(mu_tp1.eq.zero)then
              !photons are confined in the equatorial plane, so the integrations about \theta are valished.
              timet=zero
              phyt=zero
              mucos=zero
              return
          endif
          !************************************************************************** 
          If(a_spin.eq.zero .or. mve.eq.one)then  
              CALL phyt_schwatz(kp,kt,lambda,q,mve,p,sin_ini,cos_ini,a_spin,phyt,timet,mucos,t1,t2)  
              return
          endif  
   
          If(kt.lt.zero)then
              PI01=-PI0 
          else
              PI01=PI0
          endif
          tmu=weierstrassP(p+PI01,g2,g3,dd,del)
          mucos = mu_tp1+b0/(four*tmu-b1)  
          !to get number of turn points of t1 and t2.
          !111111111*****************************************************************************************
          !mu=mu_tp+b0/(four*tmu-b1) 
          index_p5(1)=0
          cases_int=1
          call weierstrass_int_J3(tobs,tmu,dd,del,a4,b4,index_p5,rff_p,integ5,cases_int)  
          pp=integ5(1)
          p1=PI0-pp
          p2=p_mt1_mt2-p1 
!========================================================================
    !======== To determine the number of time_0 N_t1, N_t2 that the particle meets the
    !======== two turn points r_tp1, r_tp2 
                      If(mobseqmtp)then
                          p1_temp = zero
                          p2_temp = p_mt1_mt2
                          t1 = 0
                          t2 = 0
                          N_temp = 0
                          If(cos_ini.eq.mu_tp1)then
                              Do While(.TRUE.)
                                  IF(p1_temp.le.p .and. p.le.p2_temp)then
                                      EXIT
                                  ELSE
                                      N_temp = N_temp + 1
                                      p1_temp = p2_temp
                                      p2_temp = p2_temp + p_mt1_mt2
                                      t1 = t2
                                      t2 = N_temp-t1
                                  ENDIF
                              ENDDO
                          Else If(cos_ini.eq.mu_tp2)then
                              Do While(.TRUE.)
                                  IF(p1_temp.le.p .and. p.le.p2_temp)then
                                      EXIT
                                  ELSE
                                      N_temp = N_temp + 1
                                      p1_temp = p2_temp
                                      p2_temp = p2_temp + p_mt1_mt2
                                      t2 = t1
                                      t1 = N_temp-t2
                                  ENDIF
                              ENDDO
                          Endif 
                      Else
                          p1_temp = zero
                          t1 = 0
                          t2 = 0
                          N_temp = 0
                          If(kt.gt.zero)then
                              p2_temp = PI2_p
                              Do While(.TRUE.)
                                  IF(p1_temp.le.p .and. p.le.p2_temp)then
                                      EXIT
                                  ELSE
                                      N_temp = N_temp + 1
                                      p1_temp = p2_temp
                                      p2_temp = p2_temp + p_mt1_mt2
                                      t1 = t2
                                      t2 = N_temp-t1
                                  ENDIF
                              ENDDO
                          ENDIF
                          If(kt.lt.zero)then
                              p2_temp = PI1_p
                              Do While(.TRUE.)
                                  IF(p1_temp.le.p .and. p.le.p2_temp)then
                                      EXIT
                                  ELSE
                                      N_temp = N_temp + 1
                                      p1_temp = p2_temp
                                      p2_temp = p2_temp + p_mt1_mt2
                                      t2 = t1
                                      t1 = N_temp-t2
                                  ENDIF
                              ENDDO
                          ENDIF
                      Endif   
!========================================================================
          index_p5(1)=-1
          index_p5(2)=-2
          index_p5(3)=0
          index_p5(4)=-4
          index_p5(5)=0
          !*****pp part***************************************
          If(lambda.ne.zero)then 
              cases_int=2
              call weierstrass_int_J3(tobs,tmu,dd,del,-a_add,b4,index_p5,abs(pp),integ5,cases_int)
              call weierstrass_int_J3(tobs,tmu,dd,del,-a_m,b4,index_p5,abs(pp),integ15,cases_int)
              pp_phi=(pp/(one-mu_tp1**two)+(integ5(2)*c_add-integ15(2)*c_m)/two)*lambda
          else 
              pp_phi=zero   
          endif
          cases_int=4
          call weierstrass_int_J3(tobs,tmu,dd,del,h,b4,index_p5,abs(pp),integ,cases_int)
          pp_t=(pp*mu_tp1**two+integ(2)*mu_tp1*b0/two+integ(4)*b0**two/sixteen)*a_spin*a_spin
          !*****p1 part***************************************
          If(t1.eq.0)then 
              p1_phi=zero
              p1_t=zero
          else  
              If(lambda.ne.zero)then  
                  IF(PI1_phi .EQ. zero)THEN
                      cases_int=2 
                      call weierstrass_int_J3(tobs,infinity,dd,del,-a_add,b4,index_p5,PI0,integ5,cases_int)
                      call weierstrass_int_J3(tobs,infinity,dd,del,-a_m,b4,index_p5,PI0,integ15,cases_int)
                      PI1_phi=(PI0/(one-mu_tp1**two)+(integ5(2)*c_add-integ15(2)*c_m)/two)*lambda
                  ENDIF 
                  p1_phi=PI1_phi-pp_phi 
              else 
                  p1_phi=zero      
              endif 
              IF(PI1_time .EQ. zero)THEN  
                  cases_int=4  
                  call weierstrass_int_J3(tobs,infinity,dd,del,h,b4,index_p5,PI0,integ,cases_int)
                  PI1_time=(PI0*mu_tp1**two+integ(2)*mu_tp1*b0/two+integ(4)*b0**two/sixteen)*a_spin**two 
              ENDIF
              p1_t=PI1_time-pp_t 
          endif 
          !*****p2 part***************************************
          If(t2.eq.0)then
              p2_phi=zero
              p2_t=zero
          else
              IF(lambda.ne.zero)then  
                  IF(PI2_phi .EQ. zero)THEN  
                      cases_int=2 
                      call weierstrass_int_J3(tp2,tobs,dd,del,-a_add,b4,index_p5,PI2_p,integ5,cases_int)
                      call weierstrass_int_J3(tp2,tobs,dd,del,-a_m,b4,index_p5,PI2_p,integ15,cases_int)
                      PI2_phi=(PI2_p/(one-mu_tp1*mu_tp1)+(integ5(2)*c_add-integ15(2)*c_m)/two)*lambda 
                  ENDIF
                  p2_phi=PI2_phi+pp_phi
              ELSE
                  p2_phi=zero                
              ENDIF 
                
              IF(PI2_time .EQ. zero)THEN  
                  cases_int=4  
                  call weierstrass_int_J3(tp2,tobs,dd,del,h,b4,index_p5,PI2_p,integ,cases_int)
                  PI2_time=(PI2_p*mu_tp1**two+integ(2)*mu_tp1*b0/two+integ(4)*b0**two/sixteen)*a_spin**two 
              ENDIF 
              p2_t=PI2_time+pp_t   
          endif   
 !**************************************************************
          !write(*,*)'kkkk=',pp_t,p1_t,p2_t
          If(mobseqmtp)then 
              If(cos_ini.eq.mu_tp1)then
                  phyt=-pp_phi+two*(t1*p1_phi+t2*p2_phi)
                  timet=-pp_t+two*(t1*p1_t+t2*p2_t) 
              else
                  phyt=pp_phi+two*(t1*p1_phi+t2*p2_phi)
                  timet=pp_t+two*(t1*p1_t+t2*p2_t)  
              endif
          else
              If(kt.lt.zero)then
                  phyt=pp_phi+two*(t1*p1_phi+t2*p2_phi) 
                  timet=pp_t+two*(t1*p1_t+t2*p2_t)
              endif
              If(kt.gt.zero)then  
                  phyt=-pp_phi+two*(t1*p1_phi+t2*p2_phi)
                  timet=-pp_t+two*(t1*p1_t+t2*p2_t) 
              endif
          endif 
       !***************************************************** 
          else
              count_num=1
              goto 30   
          endif    
      ENDIF
      !write(*,*)'ff1=',phyt,timet,pp_phi,p1_phi,p2_phi,t1,t2
      RETURN
      END SUBROUTINE INTTPART


!********************************************************************************************
      SUBROUTINE phyt_schwatz(kp,kt,lambda,q,mve,p,sin_ini,cos_ini,a_spin,&
                              phyc_schwatz,timet,mucos,t1,t2)
!********************************************************************************************************
!*     PURPOSE:  Computes \mu part of integrals in coordinates \phi, t and proper time \sigma,
!*               expressed in Table 5 and 6 in Yang & Wang (2013), A&A for a_spin = 0 or mve = 1.    
!*     INPUTS:   p--------------independent variable, which must be nonnegative.
!*               kp-------------k_{\phi}, initial \phi component of four momentum of a particle
!*                              measured under an LNRF.
!*               kt-------------k_{\theta}, initial \theta component of four momentum of a particle
!*                              measured under an LNRF. 
!*               lambda,q,mve-------constants of motion, defined by lambda=L_z/E, q=Q/E^2, mve = \mu_m/E.
!*               sin_ini,cos_ini-------sin_ini=sin(\theta_{ini}), cos_ini=cos(\theta_{ini}), where 
!*                                     \theta_{ini} is the initial \theta coordinate of the particle.
!*               a_spin---------spin of black hole, on interval [-1,1].        
!*     OUTPUTS:  phyc_schwatz-----------value of integral \phi_\theta.
!*               timet----------value of integral \t_\theta. And \sigma_\theta=time_\theta.
!*               mucos----------value of function \mu(p).
!*               t1,t2----------number of time_0 of photon meets turning points \mu_tp1 and \mu_tp2
!*                              respectively for a given p.            
!*     ROUTINES CALLED: mutp, root3, weierstrass_int_J3 
!*     ACCURACY:   Machine.    
!*     AUTHOR:     Yang & Wang (2012)  
!*     DATE WRITTEN:  5 Jan 2012
!*     REVISIONS: ***************************************************************************************
      USE constants
      implicit none
      Double precision phyc_schwatz,kp,kt,p,sin_ini,cos_ini,phyr_schwatz,pp,p1,p2,a_spin,mu,&
             lambda,AA,BB,AAi,AAim,q,mu1,mu2,mu_tp1,mu_tp2,mve,timet,PI1_time,PI2_time,&
             mu2p,mucos,Pt,PI1_phi,PI2_phi,kp_1,kt_1,lambda_1,q_1,pp_time,p1_time,&
             sin_ini_1,cos_ini_1,a_spin_1,pp_phi,p1_phi,p2_phi,p2_time,&
             PI1,PI2,p1_temp,p2_temp
      integer  :: t1,t2,cases,i,j,N_temp,count_num=1 
      logical :: err,mobseqmtp
      save :: PI1,PI1_phi,PI2_phi,Pt,kp_1,kt_1,lambda_1,q_1,pp_phi,p1_phi,p2_phi,&
              sin_ini_1,cos_ini_1,a_spin_1,mobseqmtp,AA,BB,PI1_time,PI2_time,&
              mu_tp1,mu_tp2 

60    continue 
      IF(count_num .EQ. 1)THEN
          kp_1=kp
          kt_1=kt
          lambda_1=lambda
          q_1=q
          cos_ini_1=cos_ini
          sin_ini_1=sin_ini
          a_spin_1=a_spin
          t1=0
          t2=0         
          mobseqmtp=.false.
          If(q.gt.zero)then 
              AA=sqrt((lambda**two+q)/q)
              BB=sqrt(q)
 !*****************************************************
              If(kt.gt.zero)then
                  mu=sin(asin(cos_ini*AA)-p*BB*AA)/AA 
              else
                  If(kt.eq.zero)then
                      mu=cos(p*AA*BB)*cos_ini
                  else         
                      mu=sin(asin(cos_ini*AA)+p*AA*BB)/AA 
                  endif 
              endif
              mucos = mu  
        !****************************************************
              If(kt.ne.zero)then
                  mu_tp1=sqrt(q/(lambda**two+q))
                  mu_tp2=-mu_tp1 
              else
                  mu_tp1=abs(cos_ini)
                  mu_tp2=-mu_tp1
                  mobseqmtp=.true.
              endif
              If(abs(cos_ini).eq.one)mobseqmtp=.true. 

              If(mu_tp1.eq.zero)then
              !photons are confined in the equatorial plane, 
              !so the integrations about !\theta are valished.
                  timet = zero
                  phyc_schwatz = zero
                  return
              endif

              !***************************************************
              PI1=(PI/two-asin(cos_ini/mu_tp1))*mu_tp1/BB 
              Pt=PI*mu_tp1/BB 
              PI2=Pt-PI1
              pp=(asin(mu/mu_tp1)-asin(cos_ini/mu_tp1))*mu_tp1/BB 
              p1=PI1-pp
              p2=Pt-p1 
              PI1_phi=zero
              PI2_phi=zero
              PI1_time=zero
              PI2_time=zero
!========================================================================
    !======== To determine the number of time_0 N_t1, N_t2 that the particle meets the
    !======== two turn points r_tp1, r_tp2 
                      If(mobseqmtp)then
                          p1_temp = zero
                          p2_temp = pt
                          t1 = 0
                          t2 = 0
                          N_temp = 0
                          If(cos_ini.eq.mu_tp1)then
                              Do While(.TRUE.)
                                  IF(p1_temp.le.p .and. p.le.p2_temp)then
                                      EXIT
                                  ELSE
                                      N_temp = N_temp + 1
                                      p1_temp = p2_temp
                                      p2_temp = p2_temp + pt
                                      t1 = t2
                                      t2 = N_temp-t1
                                  ENDIF
                              ENDDO
                          Else If(cos_ini.eq.mu_tp2)then
                              Do While(.TRUE.)
                                  IF(p1_temp.le.p .and. p.le.p2_temp)then
                                      EXIT
                                  ELSE
                                      N_temp = N_temp + 1
                                      p1_temp = p2_temp
                                      p2_temp = p2_temp + pt
                                      t2 = t1
                                      t1 = N_temp-t2
                                  ENDIF
                              ENDDO
                          Endif 
                      Else
                          p1_temp = zero
                          t1 = 0
                          t2 = 0
                          N_temp = 0
                          If(kt.gt.zero)then
                              p2_temp = PI2
                              Do While(.TRUE.)
                                  IF(p1_temp.le.p .and. p.le.p2_temp)then
                                      EXIT
                                  ELSE
                                      N_temp = N_temp + 1
                                      p1_temp = p2_temp
                                      p2_temp = p2_temp + pt
                                      t1 = t2
                                      t2 = N_temp-t1
                                  ENDIF
                              ENDDO
                          ENDIF
                          If(kt.lt.zero)then
                              p2_temp = PI1
                              Do While(.TRUE.)
                                  IF(p1_temp.le.p .and. p.le.p2_temp)then
                                      EXIT
                                  ELSE
                                      N_temp = N_temp + 1
                                      p1_temp = p2_temp
                                      p2_temp = p2_temp + pt
                                      t2 = t1
                                      t1 = N_temp-t2
                                  ENDIF
                              ENDDO
                          ENDIF
                      Endif   
!======================================================================== 
              If(lambda.eq.zero)then  
                  pp_phi = zero
              else
                  pp_phi=lambda*schwatz_int(cos_ini,mu,AA)/BB
              endif
              If(a_spin.eq.zero)then
                  pp_time = zero
              else
                  pp_time = a_spin*a_spin*mveone_int(cos_ini,mu,one/AA)/BB
              endif
              If(t1.eq.0)then
                  p1_phi=zero
                  p1_time = zero
              else
                  IF(PI1_phi .eq. zero)THEN
                      PI1_phi = lambda*schwatz_int(cos_ini,mu_tp1,AA)/BB
                  ENDIF
                  IF(PI1_time .eq. zero .and. a_spin .ne. zero)THEN
                      PI1_time = a_spin*a_spin*mveone_int(cos_ini,mu_tp1,one/AA)/BB
                  ENDIF
                  p1_phi=PI1_phi-pp_phi    
                  p1_time=PI1_time-pp_time   
              endif
              If(t2.eq.0)then
                  p2_phi=zero
                  p2_time=zero
              else
                  IF(PI2_phi .EQ. zero)THEN
                      PI2_phi=lambda*schwatz_int(mu_tp2,cos_ini,AA)/BB
                  ENDIF
                  IF(PI2_time .eq. zero .and. a_spin.ne.zero)THEN
                      PI2_time = a_spin*a_spin*mveone_int(mu_tp2,cos_ini,one/AA)/BB
                  ENDIF
                  p2_phi=PI2_phi+pp_phi 
                  p2_time=PI2_time+pp_time
              endif
              If(mobseqmtp)then
                  If(cos_ini.eq.mu_tp1)then  
                      phyc_schwatz=-pp_phi+two*(t1*p1_phi+t2*p2_phi)
                      timet = -pp_time+two*(t1*p1_time+t2*p2_time)  
                  else
                      phyc_schwatz=pp_phi+two*(t1*p1_phi+t2*p2_phi) 
                      timet = pp_time+two*(t1*p1_time+t2*p2_time)  
                  endif  
              else
                 If(kt.lt.zero)then
                      phyc_schwatz=pp_phi+two*(t1*p1_phi+t2*p2_phi)
                      timet = pp_time+two*(t1*p1_time+t2*p2_time) 
                 endif  
                 If(kt.gt.zero)then
                      phyc_schwatz=-pp_phi+two*(t1*p1_phi+t2*p2_phi)
                      timet = -pp_time+two*(t1*p1_time+t2*p2_time) 
                 endif   
              endif
          else
              !write(unit=6,fmt=*)'phyt_schwatz(): q<0, which is a affending',&
              !  'value, the program should be',&  
              !  'stoped! and q = ',q
              !stop
              mucos=cos_ini 
              t1 = 0
              t2 = 0
              phyc_schwatz = zero
              timet=zero
          endif 
          count_num=count_num+1
      ELSE
          IF(kp_1.eq.kp.and.kt_1.eq.kt.and.lambda_1.eq.lambda.and.q_1.eq.q.and.sin_ini_1.eq.sin_ini&
          .and.cos_ini_1.eq.cos_ini.and.a_spin_1.eq.a_spin)THEN
              If(q.gt.zero)then 
         !*****************************************************
                  If(kt.gt.zero)then
                      mu=sin(asin(cos_ini*AA)-p*BB*AA)/AA 
                  else
                      If(kt.eq.zero)then
                          mu=cos(p*AA*BB)*cos_ini
                      else         
                          mu=sin(asin(cos_ini*AA)+p*AA*BB)/AA 
                      endif 
                  endif
                  mucos = mu  
        !****************************************************  
                  If(mu_tp1.eq.zero)then
                  !photons are confined in the equatorial plane, 
                  !so the integrations about !\theta are valished.
                      phyc_schwatz=zero
                      return
                  endif

                  !***************************************************  
                  pp=(asin(mu/mu_tp1)-asin(cos_ini/mu_tp1))*mu_tp1/BB 
                  p1=PI1-pp
                  p2=Pt-p1  
    !======== To determine the number of time_0 N_t1, N_t2 that the particle meets the
    !======== two turn points r_tp1, r_tp2 
                      If(mobseqmtp)then
                          p1_temp = zero
                          p2_temp = pt
                          t1 = 0
                          t2 = 0
                          N_temp = 0
                          If(cos_ini.eq.mu_tp1)then
                              Do While(.TRUE.)
                                  IF(p1_temp.le.p .and. p.le.p2_temp)then
                                      EXIT
                                  ELSE
                                      N_temp = N_temp + 1
                                      p1_temp = p2_temp
                                      p2_temp = p2_temp + pt
                                      t1 = t2
                                      t2 = N_temp-t1
                                  ENDIF
                              ENDDO
                          Else If(cos_ini.eq.mu_tp2)then
                              Do While(.TRUE.)
                                  IF(p1_temp.le.p .and. p.le.p2_temp)then
                                      EXIT
                                  ELSE
                                      N_temp = N_temp + 1
                                      p1_temp = p2_temp
                                      p2_temp = p2_temp + pt
                                      t2 = t1
                                      t1 = N_temp-t2
                                  ENDIF
                              ENDDO
                          Endif 
                      Else
                          p1_temp = zero
                          t1 = 0
                          t2 = 0
                          N_temp = 0
                          If(kt.gt.zero)then
                              p2_temp = PI2
                              Do While(.TRUE.)
                                  IF(p1_temp.le.p .and. p.le.p2_temp)then
                                      EXIT
                                  ELSE
                                      N_temp = N_temp + 1
                                      p1_temp = p2_temp
                                      p2_temp = p2_temp + pt
                                      t1 = t2
                                      t2 = N_temp-t1
                                  ENDIF
                              ENDDO
                          ENDIF
                          If(kt.lt.zero)then
                              p2_temp = PI1
                              Do While(.TRUE.)
                                  IF(p1_temp.le.p .and. p.le.p2_temp)then
                                      EXIT
                                  ELSE
                                      N_temp = N_temp + 1
                                      p1_temp = p2_temp
                                      p2_temp = p2_temp + pt
                                      t2 = t1
                                      t1 = N_temp-t2
                                  ENDIF
                              ENDDO
                          ENDIF
                      Endif  
!=================================================================================== 
                  If(lambda.eq.zero)then 
                      pp_phi = zero 
                  else
                      pp_phi=lambda*schwatz_int(cos_ini,mu,AA)/BB             
                  endif
                  If(a_spin.eq.zero)then
                      pp_time = zero
                  else
                      pp_time = a_spin*a_spin*mveone_int(cos_ini,mu,one/AA)/BB
                  endif

                  If(t1.eq.0)then
                      p1_phi=zero
                      p1_time=zero
                  else
                      IF(PI1_phi .eq. zero)THEN
                          PI1_phi = lambda*schwatz_int(cos_ini,mu_tp1,AA)/BB
                      ENDIF
                      IF(PI1_time .eq. zero .and. a_spin .ne. zero)THEN
                          PI1_time = a_spin*a_spin*mveone_int(cos_ini,mu_tp1,one/AA)/BB
                      ENDIF
                      p1_phi=PI1_phi-pp_phi    
                      p1_time=PI1_time-pp_time      
                  endif
                  If(t2.eq.0)then
                      p2_phi=zero
                      p2_time=zero
                  else
                      IF(PI2_phi .EQ. zero)THEN
                          PI2_phi=lambda*schwatz_int(mu_tp2,cos_ini,AA)/BB
                      ENDIF
                      IF(PI2_time .eq. zero .and. a_spin .ne. zero)THEN
                          PI2_time = a_spin*a_spin*mveone_int(mu_tp2,cos_ini,one/AA)/BB
                      ENDIF
                      p2_phi=PI2_phi+pp_phi    
                      p2_time=PI2_time+pp_time   
                  endif
                  If(mobseqmtp)then 
                      If(cos_ini.eq.mu_tp1)then  
                          phyc_schwatz=-pp_phi+two*(t1*p1_phi+t2*p2_phi)
                          timet = -pp_time+two*(t1*p1_time+t2*p2_time)   
                      else
                          phyc_schwatz=pp_phi+two*(t1*p1_phi+t2*p2_phi) 
                          timet = pp_time+two*(t1*p1_time+t2*p2_time)  
                      endif     
                  else
                      If(kt.lt.zero)then
                          phyc_schwatz=pp_phi+two*(t1*p1_phi+t2*p2_phi)
                          timet = pp_time+two*(t1*p1_time+t2*p2_time) 
                      endif  
                      If(kt.gt.zero)then
                          phyc_schwatz=-pp_phi+two*(t1*p1_phi+t2*p2_phi)
                          timet = -pp_time+two*(t1*p1_time+t2*p2_time) 
                      endif   
                  endif
              else
                  !write(unit=6,fmt=*)'phyt_schwatz(): q<0, which is a affending',&
                  !  'value, the program should be',&  
                  !  'stoped! and q = ',q
                  !stop
                  mucos=cos_ini 
                  t1 = 0
                  t2 = 0
                  phyc_schwatz = zero
                  timet=zero
              endif           
          ELSE 
              count_num=1
              goto 60
          ENDIF
      ENDIF    
      return
      End SUBROUTINE phyt_schwatz


!************************************************************************* 
      Function schwatz_int(y,x,AA)
!************************************************************************* 
!*     PURPOSE:  Computes \int^x_y dt/(1-t^2)/sqrt(1-AA^2*t^2) and AA .gt. 1  
!*     INPUTS:   components of above integration.      
!*     OUTPUTS:  valve of integral.             
!*     ROUTINES CALLED: NONE.
!*     ACCURACY:   Machine.    
!*     AUTHOR:     Yang & Wang (2012)  
!*     DATE WRITTEN:  5 Jan 2012
!*     REVISIONS: ******************************************  
      USE constants
      implicit none
      Double precision y,x,yt,xt,AA,schwatz_int,ppx,ppy,A2,tp  
      logical :: inverse

      xt=x
      yt=y 
      If(yt.eq.xt)then
          schwatz_int=0.D0
          return 
      endif
      If(abs(AA).ne.one)then
          A2=AA*AA
          ppx=atan(sqrt(A2-one)*xt/sqrt(abs(one-A2*xt*xt)))
          ppy=atan(sqrt(A2-one)*yt/sqrt(abs(one-A2*yt*yt)))
          schwatz_int=(ppx-ppy)/sqrt(A2-one) 
      ELse
          If(abs(xt).eq.one)then
              schwatz_int=infinity
          Else
              If(abs(yt).eq.one)then
                  schwatz_int=-infinity
              Else
                  ppx=xt/sqrt(abs(one-xt*xt))
                  ppy=yt/sqrt(abs(one-yt*yt))
                  schwatz_int=ppx-ppy 
              endif 
          Endif    
      Endif
      return
      End Function schwatz_int 


!************************************************************************* 
      FUNCTION mveone_int(x,y,z0)
!************************************************************************* 
!*    Purpose-----To compute integral \int^y_x z^2/sqrt(1-(z/z0)^2)dz, and z0<1.  
!*     INPUTS:   components of above integration.      
!*     OUTPUTS:  valve of integral.             
!*     ROUTINES CALLED: NONE.
!*     ACCURACY:   Machine.    
!*     AUTHOR:     Yang & Wang (2012)  
!*     DATE WRITTEN:  5 Jan 2012
!*     REVISIONS: ******************************************   
      USE constants
      implicit none
      double precision x,y,z0,mveone_int,xt,yt,Iy,Ix

      xt = x
      yt = y
      If(xt .eq. yt)THEN
          mveone_int = zero
          return
      ENDIF 
      Iy = z0**three*half*(dasin(yt/z0)-y/z0/z0*dsqrt(z0*z0-y*y))
      Ix = z0**three*half*(dasin(xt/z0)-x/z0/z0*dsqrt(z0*z0-x*x))
      mveone_int = Iy-Ix  
      return
      END FUNCTION mveone_int
 

!********************************************************************************************
      SUBROUTINE Int_r_Delta(z1,z2,dd,del,trp,trm,b4,h,index_p5,P,rp,rm,r_tp1,Dtp,Dtm,Atp,Atm,&
                             Dpp,Dpm,App,Apm,k1,k2,kp1,kp2,a_spin,cos_ini,lambda,b0,&
                             kvect,int_phi,int_time)
!******************************************************************************************** 
!*     Purpose-----To compute the integral: I_r = \int^r2_r1 (k1*r+k2)/D/sqrt[ R(r) ]dz, where
!*                 D = r^2-2r+a^2+e^2. I_r appears in coordinates t(p), \phi(p), and \sigma(p).
!*                 If e=0 and a=1, or a^2+e^2 = 1, D=(r-r_p)^2, r_p is the radius of the event
!*                 horizon. 
!*     INPUTS:   components of above integration.      
!*     OUTPUTS:  valve of the integral I_r.             
!*     ROUTINES CALLED: weierstrass_int_J3.
!*     ACCURACY:   Machine.    
!*     AUTHOR:     Yang & Wang (2013)  
!*     DATE WRITTEN:  5 Jan 2013
!*     REVISIONS: ******************************************   
      USE constants
      IMPLICIT NONE
      DOUBLE PRECISION z1,z2,trp,trm,b4,h,P,integ5(5),integ15(5),rp,rm,r_tp1,&
                       Dtp,Dtm,Atp,Atm,Dpp,Dpm,App,Apm,a_spin,int_phi,int_time,&
                       kp1,kp2,k1,k2,cos_ini,kvect,lambda,b0,A11,A22,A33
       
      Complex*16 dd(1:3)
      integer :: index_p5(5),cases_int,del

!======================================================================================================
      If((rp-rm)*(r_tp1-rp)*(r_tp1-rm).NE.zero)then 
          cases_int=2 
          call weierstrass_int_J3(z1,z2,dd,del,-trp,b4,index_p5,abs(P),integ5,cases_int)
          call weierstrass_int_J3(z1,z2,dd,del,-trm,b4,index_p5,abs(P),integ15,cases_int) 
          int_time = -Dtp*integ5(2)+Dtm*integ15(2)+ P*(Atp-Atm)

          IF(a_spin.NE.zero)THEN
              int_phi=P*a_spin*(App-Apm)-a_spin*(Dpp*integ5(2)-Dpm*integ15(2))
          ELSE
              int_phi=zero
          ENDIF
          IF(cos_ini.eq.zero.and.kvect.eq.zero)THEN
              int_phi=int_phi+P*lambda
          ENDIF
      Else
          If(r_tp1.eq.rp .and. rp.ne.rm)then
              cases_int=2 
              call weierstrass_int_J3(z1,z2,dd,del,-trm,b4,index_p5,abs(P),integ15,cases_int) 
              cases_int=3 
              call weierstrass_int_J3(z1,z2,dd,del,h,b4,index_p5,abs(P),integ5,cases_int)
              int_time=(k1*rp+k2)*four/(rp-rm)/b0*integ5(3)+Dtm*integ15(2)-P*Atm

              IF(a_spin.NE.zero)THEN
                  int_phi=-P*a_spin*Apm+a_spin*Dpm*integ15(2)+(kp1*rp+kp2)*four/(rp-rm)/b0*integ5(3)
              ELSE
                  int_phi=zero
              ENDIF
              IF(cos_ini.eq.zero.and.kvect.eq.zero)THEN
                  int_phi=int_phi+P*lambda
              ENDIF
          Endif  
          If(r_tp1.eq.rm .and. rp.ne.rm)then
              cases_int=2 
              call weierstrass_int_J3(z1,z2,dd,del,-trp,b4,index_p5,abs(P),integ5,cases_int) 
              cases_int=3 
              call weierstrass_int_J3(z1,z2,dd,del,h,b4,index_p5,abs(P),integ15,cases_int)
              int_time = -(k1*rm+k2)*four/(rp-rm)/b0*integ15(3)-Dtp*integ5(2)+P*Atp

              IF(a_spin.NE.zero)THEN
                  int_phi=P*a_spin*App-a_spin*Dpp*integ5(2)-(kp1*rm+kp2)*four/(rp-rm)/b0*integ15(3)
              ELSE
                  int_phi=zero
              ENDIF
              IF(cos_ini.eq.zero.and.kvect.eq.zero)THEN
                  int_phi=int_phi+P*lambda
              ENDIF
          Endif  
          If(rp.eq.rm)then !a^2+e^2=1
              If(rp.ne.r_tp1)then
                  cases_int=4 
                  call weierstrass_int_J3(z1,z2,dd,del,-trp,b4,index_p5,abs(P),integ5,cases_int) 
                  A11 = k1/(r_tp1-rp)+(k1*rp+k2)/(r_tp1-rp)**two
                  A22 = k1*b0/four/(r_tp1-rp)**two+(k1*rp+k2)*b0/two/(r_tp1-rp)**three
                  A33 = b0*b0*(k1*rp+k2)/16.D0/(r_tp1-rp)**four

                  int_time = A11*P-A22*integ5(2)+A33*integ5(4) 

                  A11 = kp1/(r_tp1-rp)+(kp1*rp+kp2)/(r_tp1-rp)**two
                  A22 = kp1*b0/four/(r_tp1-rp)**two+(kp1*rp+kp2)*b0/two/(r_tp1-rp)**three
                  A33 = b0*b0*(kp1*rp+kp2)/16.D0/(r_tp1-rp)**four
               
                  IF(a_spin.NE.zero)THEN
                      int_phi=A11*P-A22*integ5(2)+A33*integ5(4)
                  ELSE
                      int_phi=zero
                  ENDIF
                  IF(cos_ini.eq.zero.and.kvect.eq.zero)THEN
                      int_phi=int_phi+P*lambda
                  ENDIF 
              Else
                  cases_int=5
                  call weierstrass_int_J3(z1,z2,dd,del,h,b4,index_p5,abs(P),integ5,cases_int)
                  int_time = k1*four/b0*integ5(3)+(k1*rp+k2)*16/b0/b0*integ5(5)
               
                  IF(a_spin.NE.zero)THEN
                      int_phi=kp1*four/b0*integ5(3)+(kp1*rp+kp2)*16/b0/b0*integ5(5)
                  ELSE
                      int_phi=zero
                  ENDIF
                  IF(cos_ini.eq.zero.and.kvect.eq.zero)THEN
                      int_phi=int_phi+P*lambda
                  ENDIF
              Endif
          Endif
      Endif  
!======================================================================================================
      RETURN  
      End SUBROUTINE Int_r_Delta


!********************************************************************************************
      SUBROUTINE INTRPART(p,kvecr,kvect,lambda,q,mve,ep,a_spin,e,r_ini,cos_ini,&
                          phyr,timer,affr,r_coord,t1,t2)
!******************************************************************************************** 
!*     PURPOSE:  Computes r part of integrals in coordinates \phi, t and proper time \sigma,
!*               expressed in Table 5 and 6 of Yang & Wang (2013), A&A.    
!*     INPUTS:   p--------------independent variable, which must be nonnegative. 
!*               kvecr-------------k_{r}, initial r component of four momentum of a particle
!*                              measured under an LNRF. See equation (93).
!*               kvect-------------k_{\theta}, initial \theta component of four momentum of a particle
!*                              measured under an LNRF. See equation (94).
!*               lambda,q,mve,ep-------constants of motion, defined by lambda=L_z/E, q=Q/E^2,
!*                                     mve = \mu_m/E, ep=epsilon/varepsilon. 
!*               a_spin,e--------------the spin and the electric charge of the black hole, 
!*                                     and -1 =< a_spin^2+e^2 <= 1 must be satisfied. 
!*               r_ini-----------------the initial radial coordinate of the particle.       
!*     OUTPUTS:  phyr-----------value of integral \phi_r .
!*               affr-----------value of integral \sigma_r .
!*               timer----------value of integral t_r.
!*               r_coord--------value of function r(p).
!*               t1,t2----------number of time_0 of photon meets turning points r_tp1 and r_tp2
!*                              respectively for a given p.            
!*     ROUTINES CALLED: root3, weierstrass_int_J3, radiustp, weierstrassP, EllipticF, carlson_doublecomplex5 
!*     ACCURACY:   Machine.    
!*     AUTHOR:     Yang & Wang (2012)  
!*     DATE WRITTEN:  5 Jan 2012
!*     REVISIONS: ****************************************** 
      USE constants
      IMPLICIT NONE
      DOUBLE PRECISION phyr,re,a,B,p,a_spin,rhorizon,q,lambda,integ,cos_ini,cos_ini_1,&
             bc,cc,dc,ec,b0,b1,b2,b3,g2,g3,tobs,tp,pp,p1,p2,PI0,p1I0,p1J1,p1J2,&
             u,v,w,L1,L2,thorizon,m2,pinf,sn,cn,dn,rp,rm,B_add,come,kp1,kp2,&
             y,x,f1,g1,h1,f2,h2,a5,b5,a4,b4,integ0,integ1,integ2,r_ini,ttp,mve,mve_1,&
             PI1,PI2,tinf,integ05(5),integ5(5),integ15(5),pp2,tp1,c_temp,k1,k2,&
             r_tp1,r_tp2,t_inf,tp2,kvecr,kvect,p_temp,PI0_obs_inf,PI0_total,PI0_obs_hori,&
             PI0_obs_tp2,PI01,timer,affr,r_coord,cr,dr,rff_p,p_t1_t2,s,ac1,bc1,cc1,&
             h,pp_time,pp_phi,pp_aff,p1_phi,p1_time,p1_aff,sqrtcome,int_time,&
             p2_phi,p2_time,p2_aff,time_temp,sqt3,p_tp1_tp2,PI2_p,PI1_p,alpha1,alpha2,&
             PI1_phi,PI2_phi,PI1_time,PI2_time,PI1_aff,PI2_aff,e,ep,e_1,ep_1,&
             Atp,Atm,Dtp,Dtm,trp,trm,App,Apm,Dpp,Dpm,ar(5),Btp,Btm,Bpp,Bpm,p1_temp,p2_temp 
      DOUBLE PRECISION kvecr_1,kvect_1,lambda_1,q_1,a_spin_1,r_ini_1
      COMPLEX*16 bb(1:4),dd(3)
      INTEGER :: reals,i,j,t1,t2,p5,p4,index_p5(5),del,cases_int,cases,count_num=1,&
                 N_temp
      LOGICAL :: r_ini_eq_rtp,indrhorizon,r1eqr2
      SAVE :: kvecr_1,kvect_1,lambda_1,q_1,a_spin_1,r_ini_1,rhorizon,cos_ini_1,&
              rp,rm,a4,b4,B_add,r_ini_eq_rtp,indrhorizon,r_tp1,r_tp2,reals,cases,bb,&
              b0,b1,b2,b3,g2,g3,tobs,thorizon,mve_1,tp2,tinf,dd,PI0,sqrtcome,&
              PI0_obs_inf,PI0_total,PI0_obs_hori,PI0_obs_tp2,del,u,v,w,L1,L2,m2,t_inf,&
              pinf,f1,g1,h1,f2,h2,b5,h,a5,cc,PI1_phi,PI2_phi,Btp,Btm,Bpp,Bpm,&
              PI1_time,PI2_time,PI1_aff,PI2_aff,PI2_p,PI1_p,come,tp1,p_tp1_tp2,r1eqr2,&
              e_1,ep_1,Atp,Atm,Dtp,Dtm,k1,k2,trp,trm,App,Apm,Dpp,Dpm,kp1,kp2 
 
      IF(p.lt.zero)THEN
          Write(*,*)'INTRPART(): p you given is <0, which is not valid and the code should be stopped.'
          Write(*,*)'Please input a positive one and try it again.'
          STOP
      ENDIF    
40    continue 
      If(count_num.eq.1)then
          kvecr_1=kvecr 
          kvect_1=kvect
          lambda_1=lambda
          q_1=q
          mve_1=mve
          ep_1 = ep
          a_spin_1=a_spin
          e_1 = e
          r_ini_1=r_ini
          cos_ini_1 = cos_ini
  !************************************************************************************ 
          rp=one+sqrt(one-a_spin**two-e*e)
          rm=one-sqrt(one-a_spin**two-e*e)
          rhorizon=rp
          k1 = eight-two*a_spin*lambda+four*(e*ep-e*e)-e*e*e*ep
          k2 = e*e*(e*e+a_spin*lambda)-two*(a_spin*a_spin+e*e)*(two+e*ep)
          kp1 = two-ep*e
          kp2 = -(e*e+a_spin*lambda)
 
          b4=one
          a4=zero 
          IF(mve .EQ. one)THEN
              CALL INTRPART_MB(p,kvecr,kvect,lambda,q,mve,ep,&
                                   a_spin,e,r_ini,cos_ini,phyr,timer,affr,r_coord)
              count_num=count_num+1
              return
          ENDIF 
          r_ini_eq_rtp=.false.
          indrhorizon=.false.
          call radiustp(kvecr,a_spin,e,r_ini,lambda,q,mve,ep,r_tp1,r_tp2,&
                                     reals,r_ini_eq_rtp,indrhorizon,r1eqr2,cases,bb)
          IF(r1eqr2)THEN 
              phyr = zero
              timer = zero
              affr = zero
              r_coord=r_ini
              count_num=count_num+1
              return
          ENDIF
          come = mve*mve-one  
          PI1_phi=zero
          PI2_phi=zero
          PI1_time=zero
          PI2_time=zero
          PI1_aff=zero
          PI2_aff=zero 
          !write(*,*)'here phir=',reals,cases,r_tp1,r_tp2
          If(reals.ne.0)then  !** R(r)=0 has real roots and turning points exists in radial r.
              ar(1)=-come 
              ar(2)=two*(mve*mve+e*ep) 
              ar(3)=-(a_spin*a_spin*come+lambda*lambda+q+e*e*(mve*mve-ep*ep)) 
              ar(4)=two*(q+(lambda-a_spin)*(lambda-a_spin)+a_spin*e*ep*(a_spin-lambda)) 
              ar(5)=-q*(a_spin*a_spin+e*e)-e*e*(a_spin-lambda)**two 

              b0 = four*r_tp1**three*ar(1)+three*ar(2)*r_tp1**two+two*ar(3)*r_tp1+ar(4)
              b1 = two*r_tp1**two*ar(1)+ar(2)*r_tp1+ar(3)/three
              b2 = four/three*r_tp1*ar(1)+ar(2)/three
              b3 = ar(1)
              g2 = three/four*(b1*b1-b0*b2)
              g3 = (three*b0*b1*b2-two*b1*b1*b1-b0*b0*b3)/16.D0
 
              tp1 = infinity
              If(r_ini-r_tp1.ne.zero)then 
                  tobs=b0/four/(r_ini-r_tp1)+b1/four
              else
                  tobs=infinity
              endif 
              If(rhorizon-r_tp1.ne.zero)then
                  thorizon=b1/four+b0/four/(rhorizon-r_tp1)
              else
                  thorizon=infinity  
              endif 
              tp2=b0/four/(r_tp2-r_tp1)+b1/four 
              tinf=b1/four      
              h=-b1/four 

              call root3(zero,-g2/four,-g3/four,dd(1),dd(2),dd(3),del)  
              App = ((two-e*ep)*rp-(e*e+a_spin*lambda))/(rp-rm)/(r_tp1-rp)
              Apm = ((two-e*ep)*rm-(e*e+a_spin*lambda))/(rp-rm)/(r_tp1-rm) 
              Dpp = ((two-e*ep)*rp-(e*e+a_spin*lambda))*b0/four/(rp-rm)/(r_tp1-rp)**two
              Dpm = ((two-e*ep)*rm-(e*e+a_spin*lambda))*b0/four/(rp-rm)/(r_tp1-rm)**two  
              Atp = (k1*rp+k2)/(rp-rm)/(r_tp1-rp)
              Atm = (k1*rm+k2)/(rp-rm)/(r_tp1-rm)  
              Dtp = (k1*rp+k2)*b0/four/(rp-rm)/(r_tp1-rp)**two
              Dtm = (k1*rm+k2)*b0/four/(rp-rm)/(r_tp1-rm)**two
              trp = b0/four/(rp-r_tp1)+b1/four
              trm = b0/four/(rm-r_tp1)+b1/four
              !write(*,*)'sskkkkkk=',tp1,tp2,tobs  
 
              index_p5(1)=0
              cases_int=1 
              call weierstrass_int_J3(tobs,infinity,dd,del,a4,b4,index_p5,rff_p,integ05,cases_int) 
              PI0=integ05(1)   
              select case(cases)
              CASE(1)
                  If(kvecr .ge. zero)then !**photon will goto infinity.
                      index_p5(1)=0
                      cases_int=1
                      call weierstrass_int_J3(tinf,tobs,dd,del,a4,b4,index_p5,rff_p,integ05,cases_int)
                      PI0_obs_inf=integ05(1)
                      If(p.lt.PI0_obs_inf)then 
                          tp=weierstrassP(p+PI0,g2,g3,dd,del)  
                          r_coord = r_tp1+b0/(four*tp-b1) 
                          pp=-p      
                      else
                          tp=tinf! !Goto infinity, far away. 
                          r_coord = infinity
                          pp=-PI0_obs_inf 
                      endif
                      t1=0
                      t2=0   
                  ELSE 
                      If(.not.indrhorizon)then
                          index_p5(1)=0
                          cases_int=1 
                          call weierstrass_int_J3(tinf,infinity,dd,del,a4,b4,index_p5,rff_p,integ15,cases_int)
                          PI0_total=PI0+integ15(1)
                          t2=0
                          If(p.le.PI0)then
                              t1=0
                              pp=p  
                              tp=weierstrassP(p-PI0,g2,g3,dd,del)
                              r_coord = r_tp1+b0/(four*tp-b1)
                          else
                              t1=1
                              PI1_p=PI0 
                              If(p.lt.PI0_total)then 
                                  tp=weierstrassP(p-PI0,g2,g3,dd,del)
                                  r_coord = r_tp1+b0/(four*tp-b1)
                                  pp=two*PI0-p
                                  p1=abs(p-PI0)
                              else 
                                  tp=tinf !Goto infinity, far away.
                                  r_coord = infinity
                                  pp=-PI0_total+two*PI0 
                                  p1=pI0_total-PI0
                              endif 
                          endif 
                      ELSE      !kvecr<0, photon will fall into black hole unless something encountered. 
                          index_p5(1)=0  
                          cases_int=1
                          call weierstrass_int_J3(tobs,thorizon,dd,del,a4,b4,index_p5,rff_p,integ05,cases_int)
                          PI0_obs_hori=integ05(1)
                          If(p.lt.PI0_obs_hori)then 
                              tp=weierstrassP(p-PI0,g2,g3,dd,del) 
                              r_coord = r_tp1+b0/(four*tp-b1)
                              pp=p      
                          else
                              tp=thorizon! !Fall into black hole.
                              r_coord = rhorizon
                              pp=PI0_obs_hori
                          endif
                          t1=0
                          t2=0   
                      ENDIF
                  ENDIF  
              CASE(2)  
                  If(.not.indrhorizon)then
                      If(kvecr.lt.zero)then
                          PI01=-PI0
                      else
                          PI01=PI0 
                      endif
                      tp=weierstrassP(p+PI01,g2,g3,dd,del)
                      r_coord = r_tp1+b0/(four*tp-b1) 
                      index_p5(1)=0
                      cases_int=1 
                      call weierstrass_int_J3(tobs,tp,dd,del,a4,b4,index_p5,rff_p,integ5,cases_int)
                      call weierstrass_int_J3(tp2,infinity,dd,del,a4,b4,index_p5,rff_p,integ15,cases_int)
                      pp=integ5(1)
                      p_tp1_tp2=integ15(1) 
                      PI2_p=p_tp1_tp2-PI0 
                      PI1_p=PI0                         
                      p1=PI0-pp
                      p2=p_tp1_tp2-p1   
                      !***************************************************** 
                      !*****************************************************  
    !======== To determine the number of time_0 N_t1, N_t2 that the particle meets the
    !======== two turn points r_tp1, r_tp2 
                      If(r_ini_eq_rtp)then
                          p1_temp = zero
                          p2_temp = p_tp1_tp2
                          t1 = 0
                          t2 = 0
                          N_temp = 0
                          If(r_ini.eq.r_tp1)then
                              Do While(.TRUE.)
                                  IF(p1_temp.le.p .and. p.le.p2_temp)then
                                      EXIT
                                  ELSE
                                      N_temp = N_temp + 1
                                      p1_temp = p2_temp
                                      p2_temp = p2_temp + p_tp1_tp2
                                      t1 = t2
                                      t2 = N_temp-t1
                                  ENDIF
                              ENDDO
                          Else If(r_ini.eq.r_tp2)then
                              Do While(.TRUE.)
                                  IF(p1_temp.le.p .and. p.le.p2_temp)then
                                      EXIT
                                  ELSE
                                      N_temp = N_temp + 1
                                      p1_temp = p2_temp
                                      p2_temp = p2_temp + p_tp1_tp2
                                      t2 = t1
                                      t1 = N_temp-t2
                                  ENDIF
                              ENDDO
                          Endif 
                      Else
                          p1_temp = zero
                          t1 = 0
                          t2 = 0
                          N_temp = 0
                          If(kvecr.gt.zero)then
                              p2_temp = PI2_p
                              Do While(.TRUE.)
                                  IF(p1_temp.le.p .and. p.le.p2_temp)then
                                      EXIT
                                  ELSE
                                      N_temp = N_temp + 1
                                      p1_temp = p2_temp
                                      p2_temp = p2_temp + p_tp1_tp2
                                      t1 = t2
                                      t2 = N_temp-t1
                                  ENDIF
                              ENDDO
                          ENDIF
                          If(kvecr.lt.zero)then
                              p2_temp = PI1_p
                              Do While(.TRUE.)
                                  IF(p1_temp.le.p .and. p.le.p2_temp)then
                                      EXIT
                                  ELSE
                                      N_temp = N_temp + 1
                                      p1_temp = p2_temp
                                      p2_temp = p2_temp + p_tp1_tp2
                                      t2 = t1
                                      t1 = N_temp-t2
                                  ENDIF
                              ENDDO
                          ENDIF
                      Endif  
                      !****************************************************   
                  else   !photon has probability to fall into black hole.
                      If(kvecr.le.zero)then
                          index_p5(1)=0
                          cases_int=1
                          call weierstrass_int_J3(tobs,thorizon,dd,del,a4,b4,index_p5,rff_p,integ05,cases_int)
                          PI0_obs_hori=integ05(1)
                          If(p.lt.PI0_obs_hori)then 
                              tp=weierstrassP(p-PI0,g2,g3,dd,del)
            !write(*,*)'here',p,tp,tobs
                              r_coord = r_tp1+b0/(four*tp-b1) 
                              pp=p       
                          else
                              tp=thorizon! !Fall into black hole.
                              r_coord = rhorizon
                              pp=PI0_obs_hori
                          endif
                          t1=0
                          t2=0
                      ELSE  !p_r>0, photon will meet the r_tp2 turning point and turn around then goto vevnt horizon.     
                          index_p5(1)=0
                          cases_int=1 
                          call weierstrass_int_J3(tp2,tobs,dd,del,a4,b4,index_p5,rff_p,integ05,cases_int)  
                          call weierstrass_int_J3(tp2,thorizon,dd,del,a4,b4,index_p5,rff_p,integ15,cases_int)
                          PI0_obs_tp2=integ05(1) 
                          PI2_p=PI0_obs_tp2
                          PI0_total=integ15(1)+PI0_obs_tp2
                          If(p.le.PI0_obs_tp2)then
                              t1=0
                              t2=0
                              pp=-p 
                              tp=weierstrassP(p+PI0,g2,g3,dd,del)
                              r_coord = r_tp1+b0/(four*tp-b1) 
                          else
                              t1=0
                              t2=1
                              If(p.lt.PI0_total)then 
                                  tp=weierstrassP(p+PI0,g2,g3,dd,del)
                                  r_coord = r_tp1+b0/(four*tp-b1)
                                  pp=p-two*PI0_obs_tp2
                                  p2=p-PI0_obs_tp2
                              else 
                                  tp=thorizon !Fall into black hole. 
                                  r_coord = rhorizon
                                  pp=PI0_total-two*PI0_obs_tp2
                                  p2=PI0_total-PI0_obs_tp2
                              endif 
                          endif 
                      ENDIF
                  ENDIF                          
              END SELECT  
            !******************************************************************  
              index_p5(1)=-1
              index_p5(2)=-2
              index_p5(3)=2
              index_p5(4)=-4
              index_p5(5)=4
              !pp part *************************************************** 
              cases_int=4
              call weierstrass_int_J3(tobs,tp,dd,del,h,b4,index_p5,abs(pp),integ5,cases_int)
              pp_aff=integ5(4)*b0**two/sixteen+integ5(2)*b0*r_tp1/two+pp*r_tp1**two  
              pp_time=integ5(2)*(two+e*ep)*b0/four+pp*((two+e*ep)*(r_tp1+two)-e*e)+pp_aff 
 
              CALL Int_r_Delta(tobs,tp,dd,del,trp,trm,b4,h,index_p5,pp,rp,rm,r_tp1,&
                          Dtp,Dtm,Atp,Atm,Dpp,Dpm,App,Apm,k1,k2,kp1,kp2,a_spin,cos_ini,lambda,b0,&
                          kvect,pp_phi,int_time) 
              pp_time = pp_time+int_time
 
              !p1 part *******************************************************
              IF(t1 .EQ. 0)THEN
                  p1_phi=ZERO
                  p1_time=ZERO
                  p1_aff=ZERO
              ELSE
                  IF(PI1_aff .EQ. zero .AND. PI1_time .EQ. zero)THEN
                      cases_int=4
                      call weierstrass_int_J3(tobs,infinity,dd,del,h,b4,index_p5,PI0,integ5,cases_int)
                      PI1_aff = integ5(4)*b0**two/sixteen+integ5(2)*b0*r_tp1/two+PI0*r_tp1**two 
                      PI1_time = integ5(2)*(two+e*ep)*b0/four+PI0*((two+e*ep)*(two+r_tp1)-e*e)+PI1_aff  

                      CALL Int_r_Delta(tobs,infinity,dd,del,trp,trm,b4,h,index_p5,PI0,rp,rm,r_tp1,&
                             Dtp,Dtm,Atp,Atm,Dpp,Dpm,App,Apm,k1,k2,kp1,kp2,a_spin,cos_ini,lambda,b0,&
                             kvect,PI1_phi,int_time) 
                      PI1_time = PI1_time + int_time
                  Endif

                  p1_aff=PI1_aff-pp_aff
                  p1_time=PI1_time-pp_time
                  P1_phi=PI1_phi-pp_phi 
              ENDIF
             !p2 part *******************************************************
              IF(t2.EQ.ZERO)THEN
                  p2_phi=ZERO
                  p2_time=ZERO
                  p2_aff=ZERO
              ELSE
                  IF(PI2_aff .EQ. zero .AND. PI2_time .EQ. zero)THEN
                      cases_int=4
                      call weierstrass_int_J3(tp2,tobs,dd,del,h,b4,index_p5,PI2_p,integ5,cases_int)
                      PI2_aff=integ5(4)*b0**two/sixteen+integ5(2)*b0*r_tp1/two+PI2_p*r_tp1**two  
                      PI2_time=integ5(2)*(two+e*ep)*b0/four+PI2_p*((two+e*ep)*(r_tp1+two)-e*e)+PI2_aff 

                      CALL Int_r_Delta(tp2,tobs,dd,del,trp,trm,b4,h,index_p5,PI2_p,rp,rm,r_tp1,&
                             Dtp,Dtm,Atp,Atm,Dpp,Dpm,App,Apm,k1,k2,kp1,kp2,a_spin,cos_ini,lambda,b0,&
                             kvect,PI2_phi,int_time) 
                      PI2_time = PI2_time + int_time
                  ENDIF
                  p2_aff=PI2_aff+pp_aff
                  p2_time=PI2_time+pp_time
                  p2_phi=PI2_phi+pp_phi
              ENDIF
              !phi, aff,time part *******************************************************
              !write(*,*)'phir=',pp_phi,p1_phi,p2_phi,t1,t2,cases
              !write(*,*)'timer=',pp_time,p1_time,p2_time,t1,t2,r_tp1,r_tp2
              If(.not.r_ini_eq_rtp)then
                  phyr=(sign(one,-kvecr)*pp_phi+two*(t1*p1_phi+t2*p2_phi))!/dsqrt(abs(come))
                  timer=(sign(one,-kvecr)*pp_time+two*(t1*p1_time+t2*p2_time))!/dsqrt(abs(come))
                  affr=(sign(one,-kvecr)*pp_aff+two*(t1*p1_aff+t2*p2_aff))!/dsqrt(abs(come))
              else
                  IF(r_ini.eq.r_tp1)THEN
                      phyr=-pp_phi+two*(t1*p1_phi+t2*p2_phi)
                      timer=-pp_time+two*(t1*p1_time+t2*p2_time)
                      affr=-pp_aff+two*(t1*p1_aff+t2*p2_aff)
                  ELSE
                      phyr=pp_phi+two*(t1*p1_phi+t2*p2_phi)
                      timer=pp_time+two*(t1*p1_time+t2*p2_time)
                      affr=pp_aff+two*(t1*p1_aff+t2*p2_aff) 
                  ENDIF
              endif  
!************************************************************************************************        
          ELSE   !equation R(r)=0 has no real roots. we use the Jacobi's elliptic 
                 !integrations and functions to compute the integrations.
              IF(mve.lt.one)THEN
                  u=real(bb(4))
                  w=abs(aimag(bb(4)))
                  v=real(bb(3))
                  s=abs(aimag(bb(3)))
                  ac1=s*s
                  bc1=w*w+s*s+(u-v)*(u-v)
                  cc1=w*w 
                  L1=(bc1+dsqrt(bc1*bc1-four*ac1*cc1))/two/ac1            
                  L2=(bc1-dsqrt(bc1*bc1-four*ac1*cc1))/two/ac1 
                  alpha1=(L1*v-u)/(L1-one)
                  alpha2=(L2*v-u)/(L2-one)
                  thorizon = dsqrt((L1-one)/(L1-L2))*(rhorizon-alpha1)/dsqrt((rhorizon-v)**2+ac1)  
                  tobs = dsqrt((L1-one)/(L1-L2))*(r_ini-alpha1)/dsqrt((r_ini-v)**2+ac1)  
                  t_inf = dsqrt((L1-one)/(L1-L2))
                  m2 = (L1-L2)/L1  

                  f1=u**2+w**2
                  g1=-two*u
                  h1=one
                  f2=s**2+v**2
                  g2=-two*v
                  h2=one
                  a5=zero
                  b5=one
                  index_p5(1)=-1
                  index_p5(2)=-2
                  index_p5(3)=2
                  index_p5(4)=-4
                  index_p5(5)=4 
                  pinf = EllipticF(tobs,m2)
                  IF(kvecr.lt.zero)THEN
                      PI0=( pinf-EllipticF(thorizon,m2) )/s/dsqrt(-come*L1)   
                      if(p.lt.PI0)then 
                          call sncndn(p*s*sqrt(-come*L1)-pinf,one-m2,sn,cn,dn)
                          y=v+(-u+v+s*(L1-L2)*sn*abs(cn))/((L1-L2)*sn**two-(L1-one))
                          r_coord = y 
                          pp=p  
                      else
                          y=rhorizon
                          r_coord = y
                          pp=PI0
                      endif  
                      x=r_ini 
                  ELSE
                      PI0=( EllipticF(t_inf,m2)-pinf )/s/dsqrt(-come*L1)
                      if(p.lt.PI0)then
                          call sncndn(p*s*sqrt(-come*L1)+pinf,one-m2,sn,cn,dn)
                          x=v+(-u+v-s*(L1-L2)*sn*abs(cn))/((L1-L2)*sn**two-(L1-one)) 
                          r_coord = x
                          pp=p
                      else
                          x=infinity 
                          r_coord = x
                          pp=PI0
                      endif
                      y=r_ini  
                  ENDIF                  
              ENDIF   
              IF(mve.gt.one)THEN
                  u=real(bb(4))
                  w=abs(aimag(bb(4)))
                  v=real(bb(3))
                  s=abs(aimag(bb(3)))
                  ac1=s*s
                  bc1=-w*w-s*s-(u-v)*(u-v)
                  cc1=w*w 
                  L1=(bc1+dsqrt(bc1*bc1-four*ac1*cc1))/two/ac1            
                  L2=(bc1-dsqrt(bc1*bc1-four*ac1*cc1))/two/ac1 
                  alpha1=(L1*v+u)/(L1+one)
                  alpha2=(L2*v+u)/(L2+one)  
                  thorizon = dsqrt( (L1-L2)/(-L2*(one+L1)) )*(rhorizon-alpha1)/dsqrt( (rhorizon-u)**2+cc1 ) 
                  tobs = dsqrt( (L1-L2)/(-L2*(one+L1)) )*(r_ini-alpha1)/dsqrt( (r_ini-u)**2+cc1 )  
                  t_inf = dsqrt( (L1-L2)/(-L2*(one+L1)) ) 
                  m2 = (L2-L1)/L2

                  f1=-u**2-w**2
                  g1=two*u
                  h1=-one
                  f2=s**2+v**2
                  g2=-two*v
                  h2=one
                  a5=zero
                  b5=one
                  index_p5(1)=-1
                  index_p5(2)=-2
                  index_p5(3)=2
                  index_p5(4)=-4
                  index_p5(5)=4
                  pinf = EllipticF(tobs,m2)    
                  c_temp = L2*(one+L1)/(L1-L2)
                  If(kvecr.lt.zero)then
                      PI0=( abs(pinf)-EllipticF(thorizon,m2) )/s/dsqrt(-come*L2) 
                      if(p.lt.PI0)then
                          call sncndn(p*s*dsqrt(-come*L2)-pinf,one-m2,sn,cn,dn)
                          y=u+(-L1*(u-v)/(one+L1)+w*sn*dsqrt(one-(sn*c_temp)**2))/(one+sn*sn*c_temp)
                          r_coord=y
                          pp=p
                      else
                          y=rhorizon
                          r_coord=y
                          pp=PI0
                      endif
                      x=r_ini      
                  else
                      PI0=( EllipticF(t_inf,m2)-abs(pinf) )/s/dsqrt(-come*L2)
                      if(p.lt.PI0)then
                          call sncndn(p*s*dsqrt(-come*L2)+pinf,one-m2,sn,cn,dn)
                          x=u+(-L1*(u-v)/(one+L1)-w*sn*dsqrt(one-(sn*c_temp)**2))/(one+sn*sn*c_temp)
                          r_coord=x
                          pp=p   
                      else
                          x=infinity
                          r_coord=x
                          pp=PI0 
                      endif
                      y=r_ini 
                  endif                    
              ENDIF   
              !affine parameter part integration **********************************************
              sqrtcome = dsqrt(abs(come))  
              cases_int=5
              call carlson_doublecomplex5m(y,x,f1,g1,h1,f2,g2,h2,a5,b5,index_p5,abs(pp)*sqrtcome,integ5,cases_int)
              affr=integ5(5)/sqrtcome
              timer=(two+e*ep)*integ5(3)+(two*(two+e*ep)-e*e)*pp 

              If(rp.ne.rm)then
                  cases_int=2
                  call carlson_doublecomplex5m(y,x,f1,g1,h1,f2,g2,h2,-rp,b5,index_p5,abs(pp)*sqrtcome,integ5,cases_int)
                  call carlson_doublecomplex5m(y,x,f1,g1,h1,f2,g2,h2,-rm,b5,index_p5,abs(pp)*sqrtcome,integ15,cases_int)
                 !phy part**************************************************************************
                  Btp = (k1*rp+k2)/(rp-rm)
                  Btm = (k1*rm+k2)/(rp-rm)   
                  Bpp = (kp1*rp+kp2)/(rp-rm)  
                  Bpm = (kp1*rm+kp2)/(rp-rm)
                  timer=(timer+Btp*integ5(2)-Btm*integ15(2))/sqrtcome+affr
                  phyr=a_spin*(Bpp*integ5(2)-Bpm*integ15(2))/sqrtcome
                  IF(cos_ini.eq.zero.and.kvect.eq.zero)THEN
                      phyr=phyr+pp*lambda
                  ENDIF  
              Else If (rp.eq.rm) then
                  cases_int=4
                  call carlson_doublecomplex5m(y,x,f1,g1,h1,f2,g2,h2,-rp,b5,index_p5,abs(pp)*sqrtcome,integ5,cases_int) 
                  timer=(timer+k1*integ5(2)+(k1*rp+k2)*integ5(4))/sqrtcome+affr
                  phyr=a_spin*(kp1*integ5(2)+(kp1*rp+kp2)*integ5(4))/sqrtcome
                  IF(cos_ini.eq.zero.and.kvect.eq.zero)THEN
                      phyr=phyr+pp*lambda
                  ENDIF   
              Endif  
          ENDIF
          count_num=count_num+1
      !*****************************************************************************************
      else 
      !*****************************************************************************************
        If(kvecr.eq.kvecr_1.and.kvect.eq.kvect_1.and.lambda.eq.lambda_1.and.q.eq.q_1.and.&
        mve.eq.mve_1.and.a_spin.eq.a_spin_1.and.r_ini.eq.r_ini_1&
        .and.e_1.eq.e.and.ep_1.eq.ep.and.cos_ini.eq.cos_ini_1)then
     !*********************************************************************** 
              IF(mve .EQ. one)THEN
                  CALL INTRPART_MB(p,kvecr,kvect,lambda,q,mve,ep,&
                                   a_spin,e,r_ini,cos_ini,phyr,timer,affr,r_coord)
                  return
              ENDIF  
          If(reals.ne.0)then  !** R(r)=0 has real roots and turning points exists in radial r.
              IF(r1eqr2)THEN 
                  phyr = zero
                  timer = zero
                  affr = zero
                  r_coord=r_ini
                  return
              ENDIF  
              select case(cases)
              CASE(1)
                  If(kvecr .ge. zero)then !**photon will goto infinity. 
                      If(p.lt.PI0_obs_inf)then 
                          tp=weierstrassP(p+PI0,g2,g3,dd,del)  
                          r_coord = r_tp1+b0/(four*tp-b1) 
                          pp=-p      
                      else
                          tp=tinf! !Goto infinity, far away. 
                          r_coord = infinity
                          pp=-PI0_obs_inf 
                      endif
                      t1=0
                      t2=0   
                  ELSE 
                      If(.not.indrhorizon)then 
                          t2=0
                          If(p.le.PI0)then
                              t1=0
                              pp=p  
                              tp=weierstrassP(p-PI0,g2,g3,dd,del)
                              r_coord = r_tp1+b0/(four*tp-b1)
                          else
                              t1=1
                              PI1_p=PI0 
                              If(p.lt.PI0_total)then 
                                  tp=weierstrassP(p-PI0,g2,g3,dd,del)
                                  r_coord = r_tp1+b0/(four*tp-b1)
                                  pp=two*PI0-p
                                  p1=abs(p-PI0)
                              else 
                                  tp=tinf !Goto infinity, far away.
                                  r_coord = infinity
                                  pp=-PI0_total+two*PI0 
                                  p1=pI0_total-PI0
                              endif 
                          endif 
                      ELSE      !kvecr<0, photon will fall into black hole unless something encountered. 
                          If(p.lt.PI0_obs_hori)then 
                              tp=weierstrassP(p-PI0,g2,g3,dd,del) 
                              r_coord = r_tp1+b0/(four*tp-b1)
                              pp=p      
                          else
                              tp=thorizon! !Fall into black hole.
                              r_coord = rhorizon
                              pp=PI0_obs_hori
                          endif
                          t1=0
                          t2=0  
                      ENDIF
                  ENDIF  
              CASE(2)
                  If(.not.indrhorizon)then
                      If(kvecr.lt.zero)then
                          PI01=-PI0
                      else
                          PI01=PI0 
                      endif
                      tp=weierstrassP(p+PI01,g2,g3,dd,del)
                      r_coord = r_tp1+b0/(four*tp-b1) 
                      index_p5(1)=0
                      cases_int=1 
                      call weierstrass_int_J3(tobs,tp,dd,del,a4,b4,index_p5,rff_p,integ5,cases_int) 
                      pp=integ5(1)  
    !======== To determine the number of time_0 N_t1, N_t2 that the particle meets the
    !======== two turn points r_tp1, r_tp2  
                      If(r_ini_eq_rtp)then
                          p1_temp = zero
                          p2_temp = p_tp1_tp2
                          t1 = 0
                          t2 = 0
                          N_temp = 0
                          If(r_ini.eq.r_tp1)then
                              Do While(.TRUE.)
                                  IF(p1_temp.le.p .and. p.le.p2_temp)then
                                      EXIT
                                  ELSE
                                      N_temp = N_temp + 1
                                      p1_temp = p2_temp
                                      p2_temp = p2_temp + p_tp1_tp2
                                      t1 = t2
                                      t2 = N_temp-t1
                                  ENDIF
                              ENDDO
                          Else If(r_ini.eq.r_tp2)then
                              Do While(.TRUE.)
                                  IF(p1_temp.le.p .and. p.le.p2_temp)then
                                      EXIT
                                  ELSE
                                      N_temp = N_temp + 1
                                      p1_temp = p2_temp
                                      p2_temp = p2_temp + p_tp1_tp2
                                      t2 = t1
                                      t1 = N_temp-t2
                                  ENDIF
                              ENDDO
                          Endif 
                      Else
                          p1_temp = zero
                          t1 = 0
                          t2 = 0
                          N_temp = 0
                          If(kvecr.gt.zero)then
                              p2_temp = PI2_p
                              Do While(.TRUE.)
                                  IF(p1_temp.le.p .and. p.le.p2_temp)then
                                      EXIT
                                  ELSE
                                      N_temp = N_temp + 1
                                      p1_temp = p2_temp
                                      p2_temp = p2_temp + p_tp1_tp2
                                      t1 = t2
                                      t2 = N_temp-t1
                                  ENDIF
                              ENDDO
                          ENDIF
                          If(kvecr.lt.zero)then 
                              p2_temp = PI1_p
                              Do While(.TRUE.)   
                                  IF(p1_temp.le.p .and. p.le.p2_temp)then 
                                      EXIT
                                  ELSE
                                      N_temp = N_temp + 1
                                      p1_temp = p2_temp
                                      p2_temp = p2_temp + p_tp1_tp2
                                      t2 = t1
                                      t1 = N_temp-t2
                                  ENDIF
                              ENDDO 
                          ENDIF
                      Endif   
         !****************************************************   
                  else   !photon has probability to fall into black hole.
                      If(kvecr.le.zero)then 
                          If(p.lt.PI0_obs_hori)then 
                              tp=weierstrassP(p-PI0,g2,g3,dd,del) 
                              r_coord = r_tp1+b0/(four*tp-b1)  
                              !write(*,*)'sss=',tobs,b1/four,b0/four/(r_ini-r_tp1),r_tp1,r_ini,b0
                              pp=p      
                          else
                              tp=thorizon! !Fall into black hole.
                              r_coord = rhorizon
                              pp=PI0_obs_hori
                          endif
                          t1=0
                          t2=0
                      ELSE  !p_r>0, photon will meet the r_tp2 turning point and turn around then goto vevnt horizon.    
                          If(p.le.PI0_obs_tp2)then
                              t1=0
                              t2=0
                              pp=-p 
                              tp=weierstrassP(p+PI0,g2,g3,dd,del)
                              r_coord = r_tp1+b0/(four*tp-b1)
                          else
                              t1=0
                              t2=1
                              If(p.lt.PI0_total)then 
                                  tp=weierstrassP(p+PI0,g2,g3,dd,del)
                                  r_coord = r_tp1+b0/(four*tp-b1)
                                  pp=p-two*PI0_obs_tp2
                                  p2=p-PI0_obs_tp2
                              else 
                                  tp=thorizon !Fall into black hole. 
                                  r_coord = rhorizon
                                  pp=PI0_total-two*PI0_obs_tp2
                                  p2=PI0_total-PI0_obs_tp2
                              endif 
                          endif 
                      ENDIF
                  ENDIF                             
              END SELECT  
            !****************************************************************** 
              index_p5(1)=-1
              index_p5(2)=-2
              index_p5(3)=2
              index_p5(4)=-4
              index_p5(5)=4
              !pp part *************************************************** 
              cases_int=4
              call weierstrass_int_J3(tobs,tp,dd,del,h,b4,index_p5,abs(pp),integ5,cases_int)
              pp_aff=integ5(4)*b0**two/sixteen+integ5(2)*b0*r_tp1/two+pp*r_tp1**two  
              pp_time=integ5(2)*(two+e*ep)*b0/four+pp*((two+e*ep)*(two+r_tp1)-e*e)+pp_aff
  
              CALL Int_r_Delta(tobs,tp,dd,del,trp,trm,b4,h,index_p5,pp,rp,rm,r_tp1,&
                          Dtp,Dtm,Atp,Atm,Dpp,Dpm,App,Apm,k1,k2,kp1,kp2,a_spin,cos_ini,lambda,b0,&
                          kvect,pp_phi,int_time) 
              pp_time = pp_time+int_time 
              !p1 part *******************************************************
              IF(t1 .EQ. 0)THEN
                  p1_phi=ZERO
                  p1_time=ZERO
                  p1_aff=ZERO
              ELSE 
                  IF(PI1_aff .EQ. zero .AND. PI1_time .EQ. zero)THEN
                      cases_int=4
                      call weierstrass_int_J3(tobs,infinity,dd,del,h,b4,index_p5,PI0,integ5,cases_int)
                      PI1_aff = integ5(4)*b0**two/sixteen+integ5(2)*b0*r_tp1/two+PI0*r_tp1**two 
                      PI1_time = integ5(2)*(two+e*ep)*b0/four+PI0*((two+e*ep)*(two+r_tp1)-e*e)+PI1_aff  

                      CALL Int_r_Delta(tobs,infinity,dd,del,trp,trm,b4,h,index_p5,PI0,rp,rm,r_tp1,&
                             Dtp,Dtm,Atp,Atm,Dpp,Dpm,App,Apm,k1,k2,kp1,kp2,a_spin,cos_ini,lambda,b0,&
                             kvect,PI1_phi,int_time) 
                      PI1_time = PI1_time + int_time
                  Endif

                  p1_aff=PI1_aff-pp_aff
                  p1_time=PI1_time-pp_time
                  P1_phi=PI1_phi-pp_phi 
              ENDIF
             !p2 part *******************************************************
              IF(t2.EQ.ZERO)THEN
                  p2_phi=ZERO
                  p2_time=ZERO
                  p2_aff=ZERO
              ELSE
                  IF(PI2_aff .EQ. zero .AND. PI2_time .EQ. zero)THEN
                      cases_int=4
                      call weierstrass_int_J3(tp2,tobs,dd,del,h,b4,index_p5,PI2_p,integ5,cases_int)
                      PI2_aff=integ5(4)*b0**two/sixteen+integ5(2)*b0*r_tp1/two+PI2_p*r_tp1**two  
                      PI2_time=integ5(2)*(two+e*ep)*b0/four+PI2_p*((two+e*ep)*(r_tp1+two)-e*e)+PI2_aff 

                      CALL Int_r_Delta(tp2,tobs,dd,del,trp,trm,b4,h,index_p5,PI2_p,rp,rm,r_tp1,&
                             Dtp,Dtm,Atp,Atm,Dpp,Dpm,App,Apm,k1,k2,kp1,kp2,a_spin,cos_ini,lambda,b0,&
                             kvect,PI2_phi,int_time) 
                      PI2_time = PI2_time + int_time
                  ENDIF

                  p2_aff=PI2_aff+pp_aff
                  p2_time=PI2_time+pp_time
                  p2_phi=PI2_phi+pp_phi
              ENDIF
              !phi, aff,time part *******************************************************
              If(.not.r_ini_eq_rtp)then
                  phyr=(sign(one,-kvecr)*pp_phi+two*(t1*p1_phi+t2*p2_phi))!/dsqrt(abs(come))
                  timer=(sign(one,-kvecr)*pp_time+two*(t1*p1_time+t2*p2_time))!/dsqrt(abs(come))
                  affr=(sign(one,-kvecr)*pp_aff+two*(t1*p1_aff+t2*p2_aff))!/dsqrt(abs(come))
                  !write(*,*)'phir2=',pp_phi,p1_phi,p2_phi,t1,t2,phyr
                 !write(*,*)'phir2=',pp_time,p1_time,p2_time,t1,t2,tobs,tp
              else
                  IF(r_ini.eq.r_tp1)THEN
                      phyr=-pp_phi+two*(t1*p1_phi+t2*p2_phi)
                      timer=-pp_time+two*(t1*p1_time+t2*p2_time)
                      affr=-pp_aff+two*(t1*p1_aff+t2*p2_aff)
                  ELSE
                      phyr=pp_phi+two*(t1*p1_phi+t2*p2_phi)
                      timer=pp_time+two*(t1*p1_time+t2*p2_time)
                      affr=pp_aff+two*(t1*p1_aff+t2*p2_aff) 
                  ENDIF
              endif 
!************************************************************************************************        
          ELSE   !equation R(r)=0 has no real roots. we use the Jacobi's elliptic 
                 !integrations and functions to compute the calculations.
              IF(mve.lt.one)THEN 
                  index_p5(1)=-1
                  index_p5(2)=-2
                  index_p5(3)=2
                  index_p5(4)=-4
                  index_p5(5)=4 
                  pinf = EllipticF(tobs,m2)
                  IF(kvecr.lt.zero)THEN    
                      if(p.lt.PI0)then 
                          call sncndn(p*s*sqrt(-come*L1)-pinf,one-m2,sn,cn,dn)
                          y=v+(-u+v+s*(L1-L2)*sn*abs(cn))/((L1-L2)*sn**two-(L1-one))
                          r_coord = y 
                          pp=p  
                      else
                          y=rhorizon
                          r_coord = y
                          pp=PI0
                      endif  
                      x=r_ini 
                  ELSE 
                      if(p.lt.PI0)then
                          call sncndn(p*s*sqrt(-come*L1)+pinf,one-m2,sn,cn,dn)
                          x=v+(-u+v-s*(L1-L2)*sn*abs(cn))/((L1-L2)*sn**two-(L1-one)) 
                          r_coord = x
                          pp=p
                      else
                          x=infinity 
                          r_coord = x
                          pp=PI0
                      endif
                      y=r_ini  
                  ENDIF                  
              ENDIF   
              IF(mve.gt.one)THEN 
                  index_p5(1)=-1
                  index_p5(2)=-2
                  index_p5(3)=2
                  index_p5(4)=-4
                  index_p5(5)=4
                  pinf = EllipticF(tobs,m2)    
                  c_temp = L2*(one+L1)/(L1-L2)
                  If(kvecr.lt.zero)then  
                      if(p.lt.PI0)then
                          call sncndn(p*s*dsqrt(-come*L2)-pinf,one-m2,sn,cn,dn)
                          y=u+(-L1*(u-v)/(one+L1)+w*sn*dsqrt(one-(sn*c_temp)**2))/(one+sn*sn*c_temp)
                          r_coord=y
                          pp=p
                      else
                          y=rhorizon
                          r_coord=y
                          pp=PI0
                      endif
                      x=r_ini      
                  else 
                      if(p.lt.PI0)then
                          call sncndn(p*s*dsqrt(-come*L2)+pinf,one-m2,sn,cn,dn)
                          x=u+(-L1*(u-v)/(one+L1)-w*sn*dsqrt(one-(sn*c_temp)**2))/(one+sn*sn*c_temp)
                          r_coord=x
                          pp=p   
                      else
                          x=infinity
                          r_coord=x
                          pp=PI0 
                      endif
                      y=r_ini 
                  endif                    
              ENDIF    
              !affine parameter part integration **********************************************
              cases_int=5
              call carlson_doublecomplex5m(y,x,f1,g1,h1,f2,g2,h2,a5,b5,index_p5,abs(pp)*sqrtcome,integ5,cases_int)
              affr=integ5(5)/sqrtcome
              timer=(two+e*ep)*integ5(3)+(two*(two+e*ep)-e*e)*pp 

              If(rp.ne.rm)then
                  cases_int=2
                  call carlson_doublecomplex5m(y,x,f1,g1,h1,f2,g2,h2,-rp,b5,index_p5,abs(pp)*sqrtcome,integ5,cases_int)
                  call carlson_doublecomplex5m(y,x,f1,g1,h1,f2,g2,h2,-rm,b5,index_p5,abs(pp)*sqrtcome,integ15,cases_int)
                 !phy part************************************************************************** 
                  timer=(timer+Btp*integ5(2)-Btm*integ15(2))/sqrtcome+affr
                  phyr=a_spin*(Bpp*integ5(2)-Bpm*integ15(2))/sqrtcome
                  IF(cos_ini.eq.zero.and.kvect.eq.zero)THEN
                      phyr=phyr+pp*lambda
                  ENDIF  
              Else If (rp.eq.rm) then
                  cases_int=4
                  call carlson_doublecomplex5m(y,x,f1,g1,h1,f2,g2,h2,-rp,b5,index_p5,abs(pp)*sqrtcome,integ5,cases_int) 
                  timer=(timer+k1*integ5(2)+(k1*rp+k2)*integ5(4))/sqrtcome+affr
                  phyr=a_spin*(kp1*integ5(2)+(kp1*rp+kp2)*integ5(4))/sqrtcome
                  IF(cos_ini.eq.zero.and.kvect.eq.zero)THEN
                      phyr=phyr+pp*lambda
                  ENDIF   
              Endif    
          ENDIF
     !***************************************************************       
          ELSE
              count_num=1
              goto 40
          endif
      endif  
      RETURN
      END SUBROUTINE INTRPART 


!********************************************************************************************
      SUBROUTINE Int_r_Delta_MB(z1,z2,dd,del,trp,trm,b4,index_p5,P,rp,rm,r_tp1,Atp,Atm,&
                             App,Apm,k1,k2,kp1,kp2,a_spin,cos_ini,lambda,b0,&
                             kvect,int_phi,int_time) 
!******************************************************************************************** 
!*     Purpose-----To compute the integral: I_r = \int^r2_r1 (k1*r+k2)/D/sqrt[ R(r) ]dz, where
!*                 D = r^2-2r+a^2+e^2. I_r appears in coordinates t(p), \phi(p), and \sigma(p).
!*                 If e=0 and a=1, or a^2+e^2 = 1, D=(r-r_p)^2, r_p is the radius of the event
!*                 horizon. And mve = 1.
!*     INPUTS:   components of above integration.      
!*     OUTPUTS:  valve of the integral I_r.             
!*     ROUTINES CALLED: weierstrass_int_J3.
!*     ACCURACY:   Machine.    
!*     AUTHOR:     Yang & Wang (2013)  
!*     DATE WRITTEN:  5 Jan 2013
!*     REVISIONS: ******************************************   
      USE constants
      IMPLICIT NONE
      DOUBLE PRECISION z1,z2,trp,trm,b4,h,P,integ5(5),integ15(5),rp,rm,r_tp1,&
                       Atp,Atm,App,Apm,a_spin,int_phi,int_time,&
                       kp1,kp2,k1,k2,cos_ini,kvect,lambda,b0,A11,A22,A33
       
      Complex*16 dd(1:3)
      integer :: index_p5(5),cases_int,del
!======================================================================================================
      If(rp.ne.rm)then         
          cases_int=2 
          call weierstrass_int_J3(z1,z2,dd,del,-trp,b4,index_p5,abs(P),integ5,cases_int)
          call weierstrass_int_J3(z1,z2,dd,del,-trm,b4,index_p5,abs(P),integ15,cases_int)
          int_time = b0/four*(Atp*integ5(2)-Atm*integ15(2))

          IF(a_spin.NE.zero)THEN   
              int_phi = a_spin*b0/four*(App*integ5(2)-Apm*integ15(2)) 
          ELSE
              int_phi = zero
          ENDIF
          IF(cos_ini.eq.zero.and.kvect.eq.zero)THEN
              int_phi = int_phi+P*lambda
          ENDIF 
      Else If(rp.eq.rm)then !a^2+e^2=1 
          cases_int=4 
          call weierstrass_int_J3(z1,z2,dd,del,-trp,b4,index_p5,abs(P),integ5,cases_int)   
          int_time = k1*b0/four*integ5(2)+(k1*rp+k2)*b0*b0/16.D0*integ5(4) 
          
          IF(a_spin.NE.zero)THEN
              int_phi = kp1*b0/four*integ5(2)+(kp1*rp+kp2)*b0*b0/16.D0*integ5(4)
          ELSE
              int_phi = zero
          ENDIF
          IF(cos_ini.eq.zero.and.kvect.eq.zero)THEN
              int_phi = int_phi+P*lambda
          ENDIF   
      Endif  
!======================================================================================================
      RETURN  
      End SUBROUTINE Int_r_Delta_MB



!**********************************************************************************
      SUBROUTINE INTRPART_MB(p,kvecr,kvect,lambda,q,mve,ep,a_spin,e,r_ini,cos_ini,phyr,timer,affr,r_coord)
!**********************************************************************************
      USE constants
      IMPLICIT NONE
      DOUBLE PRECISION :: p,kvecr,kvect,lambda,q,mve,a_spin,r_ini,cos_ini,&
             phyr,timer,affr,r_coord,integ05(5),integ5(5),integ15(5),&
             b0,b1,b2,b3,g2,g3,z_ini,z_horizon,tinf,rhorizon,rff_p,a4,b4,&
             PI0_obs_inf,z_tp1,z_tp2,PI0,PI0_total,PI1_p,PI2_p,PI1_phi,PI2_phi,&
             PI1_time,PI2_time,PI1_aff,PI2_aff,rp,rm,k1,k2,e1,e2,e3,half_period,&
             kvecr_1,kvect_1,lambda_1,q_1,mve_1,a_spin_1,r_ini_1,cos_ini_1,&
             wp,wm,hp,hm,wbarp,wbarm,pp,p1,p2,PI01,PI2,p_temp,h,pp_aff,&
             pp_time,pp_phi,time_temp,E_add,E_m,p1_phi,p1_time,p1_aff,p2_phi,&
             p2_time,p2_aff,ac1,bc1,cc1,e,ep,Atp,Atm,App,Apm,PI0_ini_hori,&
             ep_1,e_1,trp,trm,PI0_ini_inf,xi_tp1,xi_tp2,xi_horizon,xi_ini,xi_p,&
             r_tp1,r_tp2,z_p,p_tp1_tp2,p1_temp,p2_temp,kp1,kp2,int_time,int_phi    
      COMPLEX*16 :: dd(3),bb(4)
      INTEGER :: del,index_p5(5),cases_int,reals,cases,count_num=1,t1,t2,&
              i,j,N_temp
      LOGICAL :: r_ini_eq_rtp,indrhorizon,r1eqr2 
      SAVE ::  kvecr_1,kvect_1,lambda_1,q_1,mve_1,a_spin_1,r_ini_1,cos_ini_1,&
               rhorizon,rp,rm,b4,a4,r_ini_eq_rtp,k1,k2,PI0_ini_inf,kp1,kp2,&
               indrhorizon,z_tp1,z_tp2,reals,cases,bb,dd,del,PI0,PI1_p,PI2_p,&
               PI0_obs_inf,PI0_total,ac1,bc1,cc1,PI1_phi,PI2_phi,p_tp1_tp2,&
               PI1_time,PI2_time,PI1_aff,PI2_aff,PI01,b0,b1,b2,b3,g2,g3,r1eqr2,&
               Atp,Atm,App,Apm,ep_1,e_1,trp,trm,e1,e2,e3,half_period,PI0_ini_hori,&
               xi_tp1,xi_tp2,xi_horizon,xi_ini,r_tp1,r_tp2,z_ini,z_horizon

70    continue
      If(count_num.eq.1)then
          kvecr_1=kvecr 
          kvect_1=kvect
          lambda_1=lambda
          q_1=q
          mve_1=mve
          ep_1=ep
          a_spin_1=a_spin
          r_ini_1=r_ini 
          cos_ini_1 = cos_ini
          e_1=e
       !**************************************  
          rp=one+sqrt(one-a_spin**two-e*e) 
          rm=one-sqrt(one-a_spin**two-e*e)
          rhorizon=rp 
          k1 = eight-two*a_spin*lambda+four*(e*ep-e*e)-e*e*e*ep
          k2 = e*e*(e*e+a_spin*lambda)-two*(a_spin*a_spin+e*e)*(two+e*ep) 
          kp1 = two-ep*e
          kp2 = e*e+a_spin*lambda

          b4=one
          a4=zero 
          r_ini_eq_rtp=.false.
          indrhorizon=.false.
          call radiustp(kvecr,a_spin,e,r_ini,lambda,q,mve,ep,r_tp1,r_tp2,&
                                     reals,r_ini_eq_rtp,indrhorizon,r1eqr2,cases,bb) 
          IF(r1eqr2)THEN 
              phyr = zero
              timer = zero
              affr = zero
              r_coord=r_ini
              return
          ENDIF
          PI1_phi=zero
          PI2_phi=zero
          PI1_time=zero
          PI2_time=zero
          PI1_aff=zero
          PI2_aff=zero
       !********************************************************************        
          b0=two*(one+e*ep)  
          b1=-(lambda*lambda+q+e*e*(one-ep*ep))/three
          b2=two*(q+(lambda-a_spin)*(lambda-a_spin)+a_spin*e*ep*(a_spin-lambda))/three
          b3=-q*(a_spin*a_spin+e*e)-e*e*(a_spin-lambda)**two
          g2 = three/four*(b1*b1-b0*b2)
          g3 = (three*b0*b1*b2-two*b1*b1*b1-b0*b0*b3)/16.D0    
       
          call root3(zero,-g2/four,-g3/four,dd(1),dd(2),dd(3),del)
          e1 = real(dd(1))
          e2 = real(dd(2))
          e3 = real(dd(3))
          half_period = halfperiodwp(g2,g3,dd,del)
 
          App = ((two-e*ep)*rp-(e*e+a_spin*lambda))/(rp-rm)
          Apm = ((two-e*ep)*rm-(e*e+a_spin*lambda))/(rp-rm)   
          Atp = (k1*rp+k2)/(rp-rm)
          Atm = (k1*rm+k2)/(rp-rm) 
          trp = b0/four*rp+b1/four
          trm = b0/four*rm+b1/four

          If(del .eq. 3)then
              z_tp1 = e3
              z_tp2 = e2
          else
              z_tp1 = e1
              z_tp2 = infinity
          endif
          z_horizon = b0/four*rhorizon+b1/four
          z_ini = b0/four*r_ini+b1/four  

!**********************select cases******************************************************
              select case(cases)
              case(1)
                  index_p5(1)=0
                  cases_int=1
                  call weierstrass_int_J3(z_ini,infinity,dd,del,a4,b4,index_p5,rff_p,integ05,cases_int) 
                  !call weierstrass_int_J3(tp1,z_ini,dd,del,a4,b4,index_p5,rff_p,integ15,cases_int) 
                  !call weierstrass_int_J3(z_ini,tp2,dd,del,a4,b4,index_p5,rff_p,integ5,cases_int) 
                  PI0=integ05(1) 
                  !PI1_p=integ15(1)  
                  !PI2_p=integ5(1) 
                  If(kvecr.ge.zero)then
                      PI01 = -PI0*sign(one,b0)
                  else
                      PI01 = PI0*sign(one,b0) 
                  endif 
                  IF(kvecr.ge.zero)THEN
                      If(b0.ge.zero)then 
                          PI0_ini_inf = PI0  
                          If(p.lt.PI0_ini_inf)then 
                              z_p = weierstrassP(p+PI01,g2,g3,dd,del)
                              r_coord = (four*z_p-b1)/b0  
                              pp = p 
                          else
                              z_p = infinity
                              r_coord = infinity !Goto infinity, far away.
                              pp = PI0_ini_inf
                          endif
                          t1 = 0
                          t2 = 0
                      else
                          call weierstrass_int_J3(z_tp1,z_ini,dd,del,a4,b4,index_p5,rff_p,integ05,cases_int)
                          call weierstrass_int_J3(z_tp1,z_horizon,dd,del,a4,b4,index_p5,rff_p,integ15,cases_int)
                          PI0_ini_hori = integ05(1)+integ15(1)
                          PI1_p = integ05(1)
                          PI2_p = integ15(1)-PI1_p
                          t2 = 0
                          If(p .lt. PI1_p)then
                              t1 = 0
                              z_p = weierstrassP(p+PI01,g2,g3,dd,del)
                              r_coord = (four*z_p-b1)/b0   
                              pp = -p
                          else
                              t1 = 1
                              If(p .lt. PI0_ini_hori)then 
                                  z_p = weierstrassP(p+PI01,g2,g3,dd,del)
                                  r_coord = (four*z_p-b1)/b0   
                                  pp = two*PI1_p-p 
                                  p1 = dabs(PI1_p-p)
                              else
                                  z_p = z_horizon
                                  r_coord = rhorizon
                                  pp = -PI2_p
                                  p1 = dabs(PI1_p-pp)
                              endif
                          endif 
                      endif
                  ELSE
                      IF(b0.ge.zero)THEN
                          IF(.NOT. indrhorizon)THEN 
                              PI0_total = two*half_period-PI0
                              PI1_p = half_period-PI0
                              PI2_p = PI0
                              t2 = 0
                              If(p .lt. PI1_p)then
                                  t1 = 0
                                  z_p = weierstrassP(p+PI01,g2,g3,dd,del)  
                                  r_coord = (four*z_p-b1)/b0 
                                  pp = -p
                              else
                                  t1 = 1
                                  If(p.lt.PI0_total)then
                                      z_p = weierstrassP(p+PI01,g2,g3,dd,del)  
                                      r_coord = (four*z_p-b1)/b0 
                                      pp=p-two*PI1_p
                                      p1=abs(p-PI1_p)
                                  else 
                                      z_p = infinity
                                      r_coord = infinity !Goto infinity, far away.
                                      pp=PI2_p
                                      p1=PI1_p+PI2_p
                                  endif  
                              endif                     
                          ELSE   !kvecr<0, photon will fall into black hole unless something encountered. 
                              index_p5(1)=0 
                              cases_int=1
                              call weierstrass_int_J3(z_horizon,z_ini,dd,del,a4,&
                                             b4,index_p5,rff_p,integ05,cases_int)
                              PI0_ini_hori=integ05(1)  
                              If(p.lt.PI0_ini_hori)then 
                                  z_p = weierstrassP(p+PI01,g2,g3,dd,del)
                                  r_coord = (four*z_p-b1)/b0 
                                  pp=-p 
                              else 
                                  z_p = z_horizon
                                  r_coord = rhorizon !Fall into black hole.
                                  pp=-PI0_ini_hori
                              endif
                              t1=0
                              t2=0  
                          ENDIF
                      ELSE
                          call weierstrass_int_J3(z_ini,z_horizon,dd,del,&
                                   a4,b4,index_p5,rff_p,integ05,cases_int) 
                          PI0_ini_hori=integ05(1)    
                          If(p.lt.PI0_ini_hori)then 
                              z_p = weierstrassP(p+PI01,g2,g3,dd,del)
                              r_coord = (four*z_p-b1)/b0 
                              pp = p 
                          else 
                              z_p = z_horizon
                              r_coord = rhorizon
                              pp = PI0_ini_hori 
                          endif  
                      ENDIF
                  ENDIF 
              case(2)
                  index_p5(1)=0 
                  cases_int=1
                  xi_ini = e2-(e1-e2)*(e2-e3)/(z_ini-e2)
                  xi_tp1 = e1 !e2-(e1-e2)*(e2-e3)/(e3-e2) ! for z_tp1 = e3, z_tp2 = e2.
                  xi_tp2 = infinity ! e2-(e1-e2)*(e2-e3)/(z_tp2-e2), for z_tp2=e2.
                  xi_horizon = e2-(e1-e2)*(e2-e3)/(z_horizon-e2)
                  call weierstrass_int_J3(xi_ini,infinity,dd,del,a4,b4,&
                                       index_p5,rff_p,integ05,cases_int)  
                  PI0=integ05(1) 
                  If(kvecr.ge.zero)then
                      PI01 = -PI0*sign(one,b0)
                  else
                      PI01 = PI0*sign(one,b0) 
                  endif 
                  If(.not.indrhorizon)then 
                      xi_p = weierstrassP(p+PI01,g2,g3,dd,del)
                      r_coord = (four*e2-b1-four*(e1-e2)*(e2-e3)/(xi_p-e2))/b0     
                      z_p = e2-(e1-e2)*(e2-e3)/(xi_p-e2)
                      call weierstrass_int_J3(xi_ini,xi_p,dd,del,a4,b4,index_p5,rff_p,integ5,cases_int) 
                      call weierstrass_int_J3(xi_tp1,xi_ini,dd,del,a4,b4,index_p5,rff_p,integ05,cases_int) 
                      call weierstrass_int_J3(xi_ini,xi_tp2,dd,del,a4,b4,index_p5,rff_p,integ15,cases_int)  
                      pp = integ5(1)    
                      PI1_p = integ05(1) 
                      PI2_p = integ15(1)
                      p_tp1_tp2 = PI1_p+PI2_p  
                      p1=PI1_p+pp
                      p2=PI2_p-pp   
!==============================================================================
                      If(r_ini_eq_rtp)then
                          p1_temp = zero
                          p2_temp = p_tp1_tp2
                          t1 = 0
                          t2 = 0
                          N_temp = 0
                          If(b0 .ge. zero)THEN
                              If(r_ini.eq.r_tp1)then
                                  Do While(.TRUE.)
                                      IF(p1_temp.le.p .and. p.le.p2_temp)then
                                          EXIT
                                      ELSE
                                          N_temp = N_temp + 1
                                          p1_temp = p2_temp
                                          p2_temp = p2_temp + p_tp1_tp2
                                          t1 = t2
                                          t2 = N_temp-t1
                                      ENDIF
                                  ENDDO
                              Else If(r_ini.eq.r_tp2)then
                                  Do While(.TRUE.)
                                      IF(p1_temp.le.p .and. p.le.p2_temp)then
                                          EXIT
                                      ELSE
                                          N_temp = N_temp + 1
                                          p1_temp = p2_temp
                                          p2_temp = p2_temp + p_tp1_tp2
                                          t2 = t1
                                          t1 = N_temp-t2
                                      ENDIF
                                  ENDDO
                              Endif 
                          ELSE
                              If(r_ini.eq.r_tp2)then
                                  Do While(.TRUE.)
                                      IF(p1_temp.le.p .and. p.le.p2_temp)then
                                          EXIT
                                      ELSE
                                          N_temp = N_temp + 1
                                          p1_temp = p2_temp
                                          p2_temp = p2_temp + p_tp1_tp2
                                          t1 = t2
                                          t2 = N_temp-t1
                                      ENDIF
                                  ENDDO
                              Else If(r_ini.eq.r_tp1)then
                                  Do While(.TRUE.)
                                      IF(p1_temp.le.p .and. p.le.p2_temp)then
                                          EXIT
                                      ELSE
                                          N_temp = N_temp + 1
                                          p1_temp = p2_temp
                                          p2_temp = p2_temp + p_tp1_tp2
                                          t2 = t1
                                          t1 = N_temp-t2
                                      ENDIF
                                  ENDDO
                              Endif 
                          ENDIF 
                      Else
                          p1_temp = zero
                          t1 = 0
                          t2 = 0
                          N_temp = 0
                          If(kvecr*sign(one,b0).gt.zero)then
                              p2_temp = PI2_p
                              Do While(.TRUE.)
                                  IF(p1_temp.le.p .and. p.le.p2_temp)then
                                      EXIT
                                  ELSE
                                      N_temp = N_temp + 1
                                      p1_temp = p2_temp
                                      p2_temp = p2_temp + p_tp1_tp2
                                      t1 = t2
                                      t2 = N_temp-t1
                                  ENDIF
                              ENDDO
                          ENDIF
                          If(kvecr*sign(one,b0).lt.zero)then 
                              p2_temp = PI1_p
                              Do While(.TRUE.)   
                                  IF(p1_temp.le.p .and. p.le.p2_temp)then 
                                      EXIT
                                  ELSE
                                      N_temp = N_temp + 1
                                      p1_temp = p2_temp
                                      p2_temp = p2_temp + p_tp1_tp2
                                      t2 = t1
                                      t1 = N_temp-t2
                                  ENDIF
                              ENDDO 
                          ENDIF
                      Endif  
!==============================================================================
                  else
                      If(b0.ge.zero)then
                          If(kvecr.le.zero)then
                              call weierstrass_int_J3(xi_horizon,xi_ini,dd,&
                                      del,a4,b4,index_p5,rff_p,integ15,cases_int) 
                              PI0_ini_hori = integ15(1)  
                              If(p.lt.PI0_ini_hori)then 
                                  xi_p = weierstrassP(p+PI01,g2,g3,dd,del)
                                  z_p = e2-(e1-e2)*(e2-e3)/(xi_p-e2)
                                  r_coord = (four*e2-b1-four*(e1-e2)*(e2-e3)/(xi_p-e2))/b0   
                                  pp = -p
                              else
                                  xi_p = xi_horizon
                                  z_p = e2-(e1-e2)*(e2-e3)/(xi_p-e2)
                                  r_coord = rhorizon !Fall into black hole.
                                  pp = -PI0_ini_hori
                              endif  
                              t1 = 0
                              t2 = 0 
                          else
                              call weierstrass_int_J3(xi_horizon,xi_tp2,dd,&
                                         del,a4,b4,index_p5,rff_p,integ15,cases_int)    
                              PI0_ini_hori = integ15(1)+PI0
                              PI2_p = PI0
                              t1 = 0
                              If(p.lt.PI2_p)then 
                                  xi_p = weierstrassP(p+PI01,g2,g3,dd,del)
                                  z_p = e2-(e1-e2)*(e2-e3)/(xi_p-e2)
                                  r_coord = (four*e2-b1-four*(e1-e2)*(e2-e3)/(xi_p-e2))/b0  
                                  pp = p
                                  t2 = 0
                              else
                                  t2 = 1
                                  If(p.lt.PI0_ini_hori)then
                                      xi_p = weierstrassP(p+PI01,g2,g3,dd,del)
                                      z_p = e2-(e1-e2)*(e2-e3)/(xi_p-e2)
                                      r_coord = (four*e2-b1-four*(e1-e2)*(e2-e3)/(xi_p-e2))/b0  
                                      pp = two*PI2_p-p
                                      p2 = p-PI2_p
                                  else
                                      xi_p = xi_horizon
                                      z_p = e2-(e1-e2)*(e2-e3)/(xi_p-e2)
                                      r_coord = rhorizon  !Fall into black hole. 
                                      pp = two*PI2_p-PI0_ini_hori
                                      p2 = PI0_ini_hori-PI2_p 
                                  endif 
                              endif  
                          endif   
                      else
                          If(kvecr.le.zero)then
                              call weierstrass_int_J3(xi_ini,xi_horizon,dd,del,&
                                         a4,b4,index_p5,rff_p,integ15,cases_int) 
                              PI0_ini_hori = integ15(1)  
                              If(p.lt.PI0_ini_hori)then 
                                  xi_p = weierstrassP(p+PI01,g2,g3,dd,del)
                                  z_p = e2-(e1-e2)*(e2-e3)/(xi_p-e2)
                                  r_coord = (four*e2-b1-four*(e1-e2)*(e2-e3)/(xi_p-e2))/b0 
                                  pp = p
                              else
                                  xi_p = xi_horizon
                                  z_p = e2-(e1-e2)*(e2-e3)/(xi_p-e2)
                                  r_coord = rhorizon !Fall into black hole.
                                  pp = PI0_ini_hori
                              endif   
                              t1 = 0
                              t2 = 0  
                          else
                              call weierstrass_int_J3(e3,z_horizon,dd,del,a4,&
                                           b4,index_p5,rff_p,integ15,cases_int)   
                              call weierstrass_int_J3(e3,z_ini,dd,del,a4,b4,&
                                              index_p5,rff_p,integ5,cases_int) 
                              PI0_ini_hori = integ15(1)+integ5(1)
                              PI1_p = integ5(1) 
                              t2 = 0
                              If(p.lt.PI1_p)then 
                                  t1 = 0
                                  xi_p = weierstrassP(p+PI01,g2,g3,dd,del)
                                  z_p = e2-(e1-e2)*(e2-e3)/(xi_p-e2)
                                  r_coord = (four*e2-b1-four*(e1-e2)*(e2-e3)/(xi_p-e2))/b0  
                                  pp = -p 
                              else
                                  t1 = 1
                                  If(p.lt.PI0_ini_hori)then
                                      xi_p = weierstrassP(p+PI01,g2,g3,dd,del)
                                      z_p = e2-(e1-e2)*(e2-e3)/(xi_p-e2)
                                      r_coord = (four*e2-b1-four*(e1-e2)*(e2-e3)/(xi_p-e2))/b0  
                                      pp = p-two*PI1_p 
                                      p1 = p-PI1_p
                                  else
                                      xi_p = xi_horizon
                                      z_p = e2-(e1-e2)*(e2-e3)/(xi_p-e2)
                                      r_coord = rhorizon !Fall into black hole.
                                      pp = PI0_ini_hori-two*PI1_p 
                                      p1 = PI0_ini_hori-PI1_p
                                  endif 
                              endif  
                          endif   
                      endif         
                  endif 
              end select  
!**********************end select cases******************************************************     
          index_p5(1)=-1
          index_p5(2)=-2
          index_p5(3)=2
          index_p5(4)=-4
          index_p5(5)=4
          !pp part *************************************************** 
          cases_int=5
          call weierstrass_int_J3(z_ini,z_p,dd,del,a4,b4,index_p5,abs(pp),integ5,cases_int)
          pp_aff=(integ5(5)*16.D0-eight*b1*integ5(3)+b1*b1*pp)/b0/b0
          pp_time=four/b0*(two+e*ep)*integ5(3)+pp*((two+e*ep)*(two-b1/b0)-e*e)+pp_aff  

          CALL Int_r_Delta_MB(z_ini,z_p,dd,del,trp,trm,b4,index_p5,pp,rp,rm,r_tp1,Atp,Atm,&
                           App,Apm,k1,k2,kp1,kp2,a_spin,cos_ini,lambda,b0,kvect,pp_phi,int_time)

          pp_time = pp_time + int_time 
        !p1 part *******************************************************
          IF(t1 .EQ. 0)THEN
              p1_phi=ZERO
              p1_time=ZERO
              p1_aff=ZERO
          ELSE
              IF(PI1_aff .EQ. zero .AND. PI1_time .EQ. zero)THEN           
                  cases_int=5
                  call weierstrass_int_J3(z_tp1,z_ini,dd,del,a4,b4,index_p5,PI1_p,integ5,cases_int)
                  PI1_aff=(integ5(5)*16.D0-eight*b1*integ5(3)+b1*b1*pp)/b0/b0 
                  PI1_time=four/b0*(two+e*ep)*integ5(3)+pp*((two+e*ep)*(two-b1/b0)-e*e)+PI1_aff  
   
                  CALL Int_r_Delta_MB(z_tp1,z_ini,dd,del,trp,trm,b4,index_p5,PI1_p,rp,rm,r_tp1,Atp,Atm,&
                                App,Apm,k1,k2,kp1,kp2,a_spin,cos_ini,lambda,b0,kvect,PI1_phi,int_time)
 
                  PI1_time = PI1_time + int_time 
              ENDIF 
              p1_aff=PI1_aff+pp_aff
              p1_time=PI1_time+pp_time
              P1_phi=PI1_phi+pp_phi 
          ENDIF
         !p2 part *******************************************************
          IF(t2.EQ.ZERO)THEN
              p2_phi=ZERO
              p2_time=ZERO
              p2_aff=ZERO
          ELSE
              IF(PI2_aff .EQ. zero .AND. PI2_time .EQ. zero)THEN 
                  cases_int=5
                  call weierstrass_int_J3(z_ini,z_tp2,dd,del,a4,b4,index_p5,PI2_p,integ5,cases_int)
                  PI2_aff=(integ5(5)*16.D0-eight*b1*integ5(3)+b1*b1*pp)/b0/b0 
                  PI2_time=four/b0*(two+e*ep)*integ5(3)+pp*((two+e*ep)*(two-b1/b0)-e*e)+PI2_aff   
     
                  CALL Int_r_Delta_MB(z_ini,z_tp2,dd,del,trp,trm,b4,index_p5,PI2_p,rp,rm,r_tp1,Atp,Atm,&
                                App,Apm,k1,k2,kp1,kp2,a_spin,cos_ini,lambda,b0,kvect,PI2_phi,int_time)
 
                  PI2_time = PI2_time + int_time  
              ENDIF
              p2_aff=PI2_aff-pp_aff
              p2_time=PI2_time-pp_time
              p2_phi=PI2_phi-pp_phi
          ENDIF
          !phi, aff,time part *******************************************************
          If(.not.r_ini_eq_rtp)then
              phyr=sign(one,b0)*sign(one,kvecr)*pp_phi+two*(t1*p1_phi+t2*p2_phi)
              timer=sign(one,b0)*sign(one,kvecr)*pp_time+two*(t1*p1_time+t2*p2_time) 
              affr=sign(one,b0)*sign(one,kvecr)*pp_aff+two*(t1*p1_aff+t2*p2_aff)
          else
              IF(r_ini.eq.r_tp1)THEN
                  phyr=sign(one,b0)*pp_phi+two*(t1*p1_phi+t2*p2_phi)
                  timer=sign(one,b0)*pp_time+two*(t1*p1_time+t2*p2_time)
                  affr=sign(one,b0)*pp_aff+two*(t1*p1_aff+t2*p2_aff)
              ELSE
                  phyr=-sign(one,b0)*pp_phi+two*(t1*p1_phi+t2*p2_phi)
                  timer=-sign(one,b0)*pp_time+two*(t1*p1_time+t2*p2_time)
                  affr=-sign(one,b0)*pp_aff+two*(t1*p1_aff+t2*p2_aff) 
              ENDIF 
          endif 
          count_num=count_num+1
      ELSE
          !*****************************************************************************************
          IF(kvecr.eq.kvecr_1.and.kvect.eq.kvect_1.and.lambda.eq.lambda_1.and.q.eq.q_1.and.&
          mve.eq.mve_1.and.a_spin.eq.a_spin_1 .and.r_ini.eq.r_ini_1.and.&
          ep_1.eq.ep.and.e_1.eq.e.and.cos_ini.eq.cos_ini_1)then
    !**********************************************************************    
              IF(r1eqr2)THEN 
                  phyr = zero
                  timer = zero
                  affr = zero
                  r_coord=r_ini
                  return
              ENDIF            
!**********************select cases****************************************************** 
              select case(cases)
              case(1)  
                  IF(kvecr.ge.zero)THEN
                      If(b0.ge.zero)then  
                          If(p.lt.PI0_ini_inf)then 
                              z_p = weierstrassP(p+PI01,g2,g3,dd,del)
                              r_coord = (four*z_p-b1)/b0  
                              pp = p
                          else
                              z_p = infinity
                              r_coord = infinity !Goto infinity, far away.
                              pp = PI0_ini_inf
                          endif
                          t1 = 0
                          t2 = 0
                      else 
                          t2 = 0
                          If(p .lt. PI1_p)then
                              t1 = 0
                              z_p = weierstrassP(p+PI01,g2,g3,dd,del)
                              r_coord = (four*z_p-b1)/b0   
                              pp = -p
                          else
                              t1 = 1
                              If(p .lt. PI0_ini_hori)then 
                                  z_p = weierstrassP(p+PI01,g2,g3,dd,del)
                                  r_coord = (four*z_p-b1)/b0   
                                  pp = two*PI1_p-p 
                                  p1 = dabs(PI1_p-p)
                              else
                                  z_p = z_horizon
                                  r_coord = rhorizon
                                  pp = -PI2_p
                                  p1 = dabs(PI1_p-pp)
                              endif
                          endif 
                      endif
                  ELSE
                      IF(b0.ge.zero)THEN
                          IF(.NOT. indrhorizon)THEN  
                              t2 = 0
                              If(p .lt. PI1_p)then
                                  t1 = 0
                                  z_p = weierstrassP(p+PI01,g2,g3,dd,del)  
                                  r_coord = (four*z_p-b1)/b0 
                                  pp = -p
                              else
                                  t1 = 1
                                  If(p.lt.PI0_total)then
                                      z_p = weierstrassP(p+PI01,g2,g3,dd,del)  
                                      r_coord = (four*z_p-b1)/b0 
                                      pp=p-two*PI1_p
                                      p1=abs(p-PI1_p)
                                  else 
                                      z_p = infinity
                                      r_coord = infinity !Goto infinity, far away.
                                      pp=PI2_p
                                      p1=PI1_p+PI2_p
                                  endif  
                              endif                     
                          ELSE   !kvecr<0, photon will fall into black hole unless something encountered.  
                              If(p.lt.PI0_ini_hori)then 
                                  z_p = weierstrassP(p+PI01,g2,g3,dd,del)
                                  r_coord = (four*z_p-b1)/b0 
                                  pp=-p 
                              else 
                                  z_p = z_horizon
                                  r_coord = rhorizon !Fall into black hole.
                                  pp=-PI0_ini_hori
                              endif
                              t1=0
                              t2=0  
                          ENDIF
                      ELSE    
                          If(p.lt.PI0_ini_hori)then 
                              z_p = weierstrassP(p+PI01,g2,g3,dd,del)
                              r_coord = (four*z_p-b1)/b0 
                              pp = p 
                          else 
                              z_p = z_horizon
                              r_coord = rhorizon
                              pp = PI0_ini_hori 
                          endif  
                      ENDIF
                  ENDIF 
              case(2) 
                  If(.not.indrhorizon)then 
                          xi_p = weierstrassP(p+PI01,g2,g3,dd,del)
                          r_coord = (four*e2-b1-four*(e1-e2)*(e2-e3)/(xi_p-e2))/b0     
                          z_p = e2-(e1-e2)*(e2-e3)/(xi_p-e2)
                          call weierstrass_int_J3(xi_ini,xi_p,dd,del,a4,b4,index_p5,rff_p,integ5,cases_int)  
                          pp = integ5(1)      
                          p1=PI1_p+pp
                          p2=PI2_p-pp   
!==============================================================================
                      If(r_ini_eq_rtp)then
                          p1_temp = zero
                          p2_temp = p_tp1_tp2
                          t1 = 0
                          t2 = 0
                          N_temp = 0
                          If(b0 .ge. zero)THEN
                              If(r_ini.eq.r_tp1)then
                                  Do While(.TRUE.)
                                      IF(p1_temp.le.p .and. p.le.p2_temp)then
                                          EXIT
                                      ELSE
                                          N_temp = N_temp + 1
                                          p1_temp = p2_temp
                                          p2_temp = p2_temp + p_tp1_tp2
                                          t1 = t2
                                          t2 = N_temp-t1
                                      ENDIF
                                  ENDDO
                              Else If(r_ini.eq.r_tp2)then
                                  Do While(.TRUE.)
                                      IF(p1_temp.le.p .and. p.le.p2_temp)then
                                          EXIT
                                      ELSE
                                          N_temp = N_temp + 1
                                          p1_temp = p2_temp
                                          p2_temp = p2_temp + p_tp1_tp2
                                          t2 = t1
                                          t1 = N_temp-t2
                                      ENDIF
                                  ENDDO
                              Endif 
                          ELSE
                              If(r_ini.eq.r_tp2)then
                                  Do While(.TRUE.)
                                      IF(p1_temp.le.p .and. p.le.p2_temp)then
                                          EXIT
                                      ELSE
                                          N_temp = N_temp + 1
                                          p1_temp = p2_temp
                                          p2_temp = p2_temp + p_tp1_tp2
                                          t1 = t2
                                          t2 = N_temp-t1
                                      ENDIF
                                  ENDDO
                              Else If(r_ini.eq.r_tp1)then
                                  Do While(.TRUE.)
                                      IF(p1_temp.le.p .and. p.le.p2_temp)then
                                          EXIT
                                      ELSE
                                          N_temp = N_temp + 1
                                          p1_temp = p2_temp
                                          p2_temp = p2_temp + p_tp1_tp2
                                          t2 = t1
                                          t1 = N_temp-t2
                                      ENDIF
                                  ENDDO
                              Endif 
                          ENDIF 
                      Else
                          p1_temp = zero
                          t1 = 0
                          t2 = 0
                          N_temp = 0
                          If(kvecr*sign(one,b0).gt.zero)then
                              p2_temp = PI2_p
                              Do While(.TRUE.)
                                  IF(p1_temp.le.p .and. p.le.p2_temp)then
                                      EXIT
                                  ELSE
                                      N_temp = N_temp + 1
                                      p1_temp = p2_temp
                                      p2_temp = p2_temp + p_tp1_tp2
                                      t1 = t2
                                      t2 = N_temp-t1
                                  ENDIF
                              ENDDO
                          ENDIF
                          If(kvecr*sign(one,b0).lt.zero)then 
                              p2_temp = PI1_p
                              Do While(.TRUE.)   
                                  IF(p1_temp.le.p .and. p.le.p2_temp)then 
                                      EXIT
                                  ELSE
                                      N_temp = N_temp + 1
                                      p1_temp = p2_temp
                                      p2_temp = p2_temp + p_tp1_tp2
                                      t2 = t1
                                      t1 = N_temp-t2
                                  ENDIF
                              ENDDO 
                          ENDIF
                      Endif  
!==============================================================================
                  else
                      If(b0.ge.zero)then
                          If(kvecr.le.zero)then  
                              If(p.lt.PI0_ini_hori)then 
                                  xi_p = weierstrassP(p+PI01,g2,g3,dd,del)
                                  z_p = e2-(e1-e2)*(e2-e3)/(xi_p-e2)
                                  r_coord = (four*e2-b1-four*(e1-e2)*(e2-e3)/(xi_p-e2))/b0   
                                  pp = -p
                              else
                                  xi_p = xi_horizon
                                  z_p = e2-(e1-e2)*(e2-e3)/(xi_p-e2)
                                  r_coord = rhorizon !Fall into black hole.
                                  pp = -PI0_ini_hori
                              endif  
                              t1 = 0
                              t2 = 0 
                          else 
                              t1 = 0
                              If(p.lt.PI2_p)then 
                                  xi_p = weierstrassP(p+PI01,g2,g3,dd,del)
                                  z_p = e2-(e1-e2)*(e2-e3)/(xi_p-e2)
                                  r_coord = (four*e2-b1-four*(e1-e2)*(e2-e3)/(xi_p-e2))/b0  
                                  pp = p
                                  t2 = 0
                              else
                                  t2 = 1
                                  If(p.lt.PI0_ini_hori)then
                                      xi_p = weierstrassP(p+PI01,g2,g3,dd,del)
                                      z_p = e2-(e1-e2)*(e2-e3)/(xi_p-e2)
                                      r_coord = (four*e2-b1-four*(e1-e2)*(e2-e3)/(xi_p-e2))/b0  
                                      pp = two*PI2_p-p
                                      p2 = p-PI2_p
                                  else
                                      xi_p = xi_horizon
                                      z_p = e2-(e1-e2)*(e2-e3)/(xi_p-e2)
                                      r_coord = rhorizon  !Fall into black hole. 
                                      pp = two*PI2_p-PI0_ini_hori
                                      p2 = PI0_ini_hori-PI2_p 
                                  endif 
                              endif  
                          endif   
                      else
                          If(kvecr.le.zero)then  
                              If(p.lt.PI0_ini_hori)then 
                                  xi_p = weierstrassP(p+PI01,g2,g3,dd,del)
                                  z_p = e2-(e1-e2)*(e2-e3)/(xi_p-e2)
                                  r_coord = (four*e2-b1-four*(e1-e2)*(e2-e3)/(xi_p-e2))/b0 
                                  pp = p
                              else
                                  xi_p = xi_horizon
                                  z_p = e2-(e1-e2)*(e2-e3)/(xi_p-e2)
                                  r_coord = rhorizon !Fall into black hole.
                                  pp = PI0_ini_hori
                              endif   
                              t1 = 0
                              t2 = 0  
                          else 
                              t2 = 0
                              If(p.lt.PI1_p)then 
                                  t1 = 0
                                  xi_p = weierstrassP(p+PI01,g2,g3,dd,del)
                                  z_p = e2-(e1-e2)*(e2-e3)/(xi_p-e2)
                                  r_coord = (four*e2-b1-four*(e1-e2)*(e2-e3)/(xi_p-e2))/b0  
                                  pp = -p 
                              else
                                  t1 = 1
                                  If(p.lt.PI0_ini_hori)then
                                      xi_p = weierstrassP(p+PI01,g2,g3,dd,del)
                                      z_p = e2-(e1-e2)*(e2-e3)/(xi_p-e2)
                                      r_coord = (four*e2-b1-four*(e1-e2)*(e2-e3)/(xi_p-e2))/b0  
                                      pp = p-two*PI1_p 
                                      p1 = p-PI1_p
                                  else
                                      xi_p = xi_horizon
                                      z_p = e2-(e1-e2)*(e2-e3)/(xi_p-e2)
                                      r_coord = rhorizon !Fall into black hole.
                                      pp = PI0_ini_hori-two*PI1_p 
                                      p1 = PI0_ini_hori-PI1_p
                                  endif 
                              endif  
                          endif   
                      endif         
                  endif 
              end select  
!**********************end select cases******************************************************  
              index_p5(1)=-1
              index_p5(2)=-2
              index_p5(3)=2
              index_p5(4)=-4
              index_p5(5)=4
          !pp part *************************************************** 
              cases_int=5
              call weierstrass_int_J3(z_ini,z_p,dd,del,a4,b4,index_p5,abs(pp),integ5,cases_int)
              pp_aff=(integ5(5)*16.D0-eight*b1*integ5(3)+b1*b1*pp)/b0/b0
              pp_time=four/b0*(two+e*ep)*integ5(3)+pp*((two+e*ep)*(two-b1/b0)-e*e)+pp_aff  
 
              CALL Int_r_Delta_MB(z_ini,z_p,dd,del,trp,trm,b4,index_p5,pp,rp,rm,r_tp1,Atp,Atm,&
                           App,Apm,k1,k2,kp1,kp2,a_spin,cos_ini,lambda,b0,kvect,pp_phi,int_time)

              pp_time = pp_time + int_time 
        !p1 part *******************************************************
              IF(t1 .EQ. 0)THEN
                  p1_phi=ZERO
                  p1_time=ZERO
                  p1_aff=ZERO
              ELSE
                  IF(PI1_aff .EQ. zero .AND. PI1_time .EQ. zero)THEN      
                      cases_int=5
                      call weierstrass_int_J3(z_tp1,z_ini,dd,del,a4,b4,index_p5,PI1_p,integ5,cases_int)
                      PI1_aff=(integ5(5)*16.D0-eight*b1*integ5(3)+b1*b1*pp)/b0/b0 
                      PI1_time=four/b0*(two+e*ep)*integ5(3)+pp*((two+e*ep)*(two-b1/b0)-e*e)+PI1_aff  
  
                      CALL Int_r_Delta_MB(z_tp1,z_ini,dd,del,trp,trm,b4,index_p5,PI1_p,rp,rm,r_tp1,Atp,Atm,&
                                App,Apm,k1,k2,kp1,kp2,a_spin,cos_ini,lambda,b0,kvect,PI1_phi,int_time)
 
                      PI1_time = PI1_time + int_time  
                  ENDIF 
                  p1_aff=PI1_aff+pp_aff
                  p1_time=PI1_time+pp_time
                  P1_phi=PI1_phi+pp_phi 
              ENDIF
        !p2 part *******************************************************
              IF(t2.EQ.ZERO)THEN
                  p2_phi=ZERO
                  p2_time=ZERO
                  p2_aff=ZERO
              ELSE
                  IF(PI2_aff .EQ. zero .AND. PI2_time .EQ. zero)THEN 
                      cases_int=5
                      call weierstrass_int_J3(z_ini,z_tp2,dd,del,a4,b4,index_p5,PI2_p,integ5,cases_int)
                      PI2_aff=(integ5(5)*16.D0-eight*b1*integ5(3)+b1*b1*pp)/b0/b0 
                      PI2_time=four/b0*(two+e*ep)*integ5(3)+pp*((two+e*ep)*(two-b1/b0)-e*e)+PI2_aff  
 
                      CALL Int_r_Delta_MB(z_ini,z_tp2,dd,del,trp,trm,b4,index_p5,PI2_p,rp,rm,r_tp1,Atp,Atm,&
                                App,Apm,k1,k2,kp1,kp2,a_spin,cos_ini,lambda,b0,kvect,PI2_phi,int_time)
 
                      PI2_time = PI2_time + int_time  
                  ENDIF
                  p2_aff=PI2_aff-pp_aff
                  p2_time=PI2_time-pp_time
                  p2_phi=PI2_phi-pp_phi
              ENDIF
         !phi, aff,time part *******************************************************
              If(.not.r_ini_eq_rtp)then
                  phyr=sign(one,b0)*sign(one,kvecr)*pp_phi+two*(t1*p1_phi+t2*p2_phi)
                  timer=sign(one,b0)*sign(one,kvecr)*pp_time+two*(t1*p1_time+t2*p2_time) 
                  affr=sign(one,b0)*sign(one,kvecr)*pp_aff+two*(t1*p1_aff+t2*p2_aff)
              else
                  IF(r_ini.eq.r_tp1)THEN
                      phyr=sign(one,b0)*pp_phi+two*(t1*p1_phi+t2*p2_phi)
                      timer=sign(one,b0)*pp_time+two*(t1*p1_time+t2*p2_time)
                      affr=sign(one,b0)*pp_aff+two*(t1*p1_aff+t2*p2_aff)
                  ELSE
                      phyr=-sign(one,b0)*pp_phi+two*(t1*p1_phi+t2*p2_phi)
                      timer=-sign(one,b0)*pp_time+two*(t1*p1_time+t2*p2_time)
                      affr=-sign(one,b0)*pp_aff+two*(t1*p1_aff+t2*p2_aff) 
                  ENDIF 
              endif  
     !**********************************************************************          
          ELSE
              count_num=1
              goto 70
          ENDIF
      ENDIF   
      RETURN 
      END SUBROUTINE INTRPART_MB 


!********************************************************************************************
      SUBROUTINE CIR_ORBIT_PTA(p,kp,kt,lambda,q,mve,ep,sin_ini,cos_ini,a_spin,e,&
                               r_ini,theta_star,phyt,timet,sigmat,mucos,t1,t2)    
!********************************************************************************************  
!*     PURPOSE:   This routine computs the four B_L coordiants r,\mu,\phi,t and affine 
!*                parameter \sigma of the spherical motion for a given p. 
!*     INPUTS:   p--------------the independent variable.
!*               kp-------------k_{\phi}, initial \phi component of four momentum of a particle
!*                              measured under an LNRF. See equation (95).
!*               kt-------------k_{\theta}, initial \theta component of four momentum of a particle
!*                              measured under an LNRF. See equation (94).
!*               lambda,q,mve,ep-------constants of motion, defined by lambda=L_z/E, q=Q/E^2,
!*                                     mve = \mu_m/E, ep=epsilon/varepsilon. 
!*               sin_ini,cos_ini-------sin_ini=sin(\theta_{ini}), cos_ini=cos(\theta_{ini}), where 
!*                                     \theta_{ini} is the initial \theta coordinate of the particle.
!*               a_spin,e--------------the spin and the electric charge of the black hole, 
!*                                     and -1 =< a_spin^2+e^2 <= 1 must be satisfied. 
!*               r_ini-----------------the initial radial coordinate of the particle, i.e., the
!*                                     radius of the spherical motion. 
!*               theta_star------------the maximum or minimum value of the \theta coordinate of the geodesic.
!*                                     which also the \theta turning point of the spherical motion.       
!*     OUTPUTS:  phyt-----------the \phi coordinat of the photon.
!*               timet----------the t coordinats of the photon.
!*               sigmat---------the affine parameter \sigma.
!*               mucos----------the \theta coordinates of the photon, and mucos=cos(\theta).  
!*               t1,t2----------number of time_0 of photon meets turning points \mu_tp1 and \mu_tp2
!*                              respectively.      
!*     ROUTINES CALLED: metricg, circ_mb, weierstrass_int_J3, weierstrassP
!*     ACCURACY:   Machine.    
!*     AUTHOR:     Yang & Wang (2013)  
!*     DATE WRITTEN:  5 Jan 2013
!*     REVISIONS: ****************************************** 
      USE constants
      IMPLICIT NONE
      Double precision phyt,timet,kp,kt,p,sin_ini,cos_ini,a_spin,lambda,q,mu_tp1,tposition,tp2,mu,tmu,p1J2,&
             bc,cc,dc,ec,b0,b1,b2,b3,g2,g3,tobs,p1,p2,pp,c_add,c_m,a_add,a_m,p1J1,come,PI1_p,&
             p1I0,a4,b4 ,delta,mu_tp2,r_ini,mutemp ,integ5(5),integ(5),rff_p,tp1,p1_temp,p2_temp,&
             integ15(5),pp2,f1234(4),PI0,integ05(5),fzero,mu2p,PI01,h,p1_t,p2_t,pp_t,p1_phi,&
             p2_phi,pp_phi,radius,mtp1,mtp2,mucos,sqt3,difference,p_mt1_mt2,PI1_sig,PI2_sig,&
             PI1_phi,PI2_phi,PI1_time,PI2_time,PI2_p,mve,bigT,p1_sig,p2_sig,pp_sig,sigmat 
      Double precision kp_1,kt_1,lambda_1,q_1,sin_ini_1,cos_ini_1,a_spin_1,r_ini_1,mve_1,ep,&
             ep_1,e,e_1,c_phi,c_time,theta_star,theta_star_1,sinmax,mumax,Omega,somiga,&
             expnu,exppsi,expmu1,expmu2,c_tau 
      integer ::  t1,t2,i,j,reals,cases,p4,index_p5(5),del,cases_int,N_temp,count_num=1
      complex*16 bb(1:4),dd(3)
      logical :: err,mobseqmtp
      save  kp_1,kt_1,lambda_1,q_1,sin_ini_1,cos_ini_1,a_spin_1,r_ini_1,a4,b4,mu_tp1,mu_tp2,reals,&
            mobseqmtp,b0,b1,b2,b3,g2,g3,dd,del,PI0,c_m,c_add,a_m,a_add,tp2,tobs,h,p_mt1_mt2,&
            PI1_phi,PI2_phi,PI1_time,PI2_time,PI2_p,mve_1,come,tp1,bigT,Delta,c_phi,c_time,&
            PI01,sinmax,mumax,theta_star_1,e_1,ep_1,Omega,c_tau,PI1_p

30    continue
      IF(count_num.eq.1)then  
          kp_1=kp
          kt_1=kt
          lambda_1=lambda
          q_1=q
          mve_1=mve 
          ep_1 = ep
          cos_ini_1=cos_ini
          sin_ini_1=sin_ini
          a_spin_1=a_spin
          e_1 = e
          r_ini_1=r_ini
          theta_star_1 = theta_star
          t1=0
          t2=0  
          IF(theta_star.ne.90.D0 .and. theta_star.ne.180.D0)THEN
              sinmax = dsin(theta_star*dtor)
              mumax = dcos(theta_star*dtor)
          Else
              IF(theta_star.eq.90.D0)THEN
                  sinmax = one
                  mumax = zero
              ENDIF
              IF(theta_star.eq.180.D0)THEN
                  sinmax = zero
                  mumax = -one
              ENDIF
          ENDIF
          mu_tp1 = abs(mumax)
          mu_tp2 = -mu_tp1
          mobseqmtp = .false.
          If(abs(mu_tp1).eq.abs(cos_ini))then
              mobseqmtp = .true.
          endif
     !***********************************************************************   
          !mobseqmtp=.false.
          !call mutp(kp,kt,sin_ini,cos_ini,a_spin,lambda,q,mve,mu_tp1,mu_tp2,reals,mobseqmtp)
          If(mu_tp1.eq.zero)then
              !photons are confined in the equatorial plane, so the integrations about \theta are valished.
              Omega = one/(r_ini**(three/two)+a_spin)
              mucos=zero
              phyt=p
              timet=phyt/Omega
              call metricg(r_ini,sin_ini,cos_ini,a_spin,e,somiga,expnu,exppsi,expmu1,expmu2)
              c_tau = dsqrt(-expnu*expnu+somiga*somiga*exppsi*exppsi-two*exppsi*somiga*Omega+&
                            exppsi*exppsi*Omega*Omega)
              sigmat=c_tau*timet
              count_num=count_num+1 
              return
          endif
     !**************************************************************************
          If(a_spin.eq.zero .OR. mve.eq.one)then
              timet=zero
              CALL circ_mb(kp,kt,lambda,q,mve,ep,p,sin_ini,cos_ini,a_spin,e,&
                           r_ini,theta_star,phyt,timet,sigmat,mucos,t1,t2) 
              count_num=count_num+1
              return
          endif
          a4=zero
          b4=one

          come = mve*mve-one  
          b0=four*a_spin**2*mu_tp1**3*come-two*mu_tp1*(a_spin**2*come+lambda**2+q)
          b1=two*a_spin**2*mu_tp1**2*come-(a_spin**2*come+lambda**2+q)/three
          b2=four/three*a_spin**2*mu_tp1*come
          b3=a_spin**2*come
          g2=three/four*(b1**2-b0*b2)
          g3=(three*b0*b1*b2-two*b1**3-b0**2*b3)/16.D0  
          call root3(zero,-g2/four,-g3/four,dd(1),dd(2),dd(3),del)

          If(cos_ini.ne.mu_tp1)then 
              tobs=b0/four/(cos_ini-mu_tp1)+b1/four
          else
              tobs=infinity
          endif
          tp1=infinity
          tp2=b0/four/(mu_tp2-mu_tp1)+b1/four
          If(mu_tp1-one.ne.zero)then
               c_m=b0/(four*(-one-mu_tp1)**2)
             c_add=b0/(four*(one-mu_tp1)**2) 
               a_m=b0/four/(-one-mu_tp1)+b1/four
             a_add=b0/four/(one-mu_tp1)+b1/four
          endif
          index_p5(1)=0
          cases_int=1
          call weierstrass_int_J3(tobs,tp1,dd,del,a4,b4,index_p5,rff_p,integ05,cases_int)
          PI0=integ05(1) 
          If(kt.lt.zero)then
              PI01=-PI0 
          else
              PI01=PI0
          endif
          tmu=weierstrassP(p+PI01,g2,g3,dd,del)
          mucos = mu_tp1+b0/(four*tmu-b1)
          h=-b1/four 
          !to get number of turn points of t1 and t2.
          !111111111*****************************************************************************************
          !mu=mu_tp+b0/(four*tmu-b1)
          call weierstrass_int_J3(tp2,tp1,dd,del,a4,b4,index_p5,rff_p,integ15,cases_int)
          call weierstrass_int_J3(tobs,tmu,dd,del,a4,b4,index_p5,rff_p,integ5,cases_int) 
          p_mt1_mt2=integ15(1)
          PI1_p=PI0
          PI2_p=p_mt1_mt2-PI0
          pp=integ5(1)
          p1=PI0-pp
          p2=p_mt1_mt2-p1
          PI1_phi=zero
          PI2_phi=zero
          PI1_sig=zero
          PI2_sig=zero
          PI1_time=zero
          PI2_time=zero  
!========================================================================
    !======== To determine the number of time_0 N_t1, N_t2 that the particle meets the
    !======== two turn points mu_tp1, mu_tp2 
          If(mobseqmtp)then
              p1_temp = zero
              p2_temp = p_mt1_mt2
              t1 = 0
              t2 = 0
              N_temp = 0
              If(cos_ini.eq.mu_tp1)then
                  Do While(.TRUE.)
                      IF(p1_temp.le.p .and. p.le.p2_temp)then
                          EXIT
                      ELSE
                          N_temp = N_temp + 1
                          p1_temp = p2_temp
                          p2_temp = p2_temp + p_mt1_mt2
                          t1 = t2
                          t2 = N_temp-t1
                      ENDIF
                  ENDDO
              Else If(cos_ini.eq.mu_tp2)then
                  Do While(.TRUE.)
                      IF(p1_temp.le.p .and. p.le.p2_temp)then
                          EXIT
                      ELSE
                          N_temp = N_temp + 1
                          p1_temp = p2_temp
                          p2_temp = p2_temp + p_mt1_mt2
                          t2 = t1
                          t1 = N_temp-t2
                      ENDIF
                  ENDDO
              Endif 
          Else
              p1_temp = zero
              t1 = 0
              t2 = 0
              N_temp = 0
              If(kt.gt.zero)then
                  p2_temp = PI2_p
                  Do While(.TRUE.)
                      IF(p1_temp.le.p .and. p.le.p2_temp)then
                          EXIT
                      ELSE
                          N_temp = N_temp + 1
                          p1_temp = p2_temp
                          p2_temp = p2_temp + p_mt1_mt2
                          t1 = t2
                          t2 = N_temp-t1
                      ENDIF
                  ENDDO
              ENDIF
              If(kt.lt.zero)then
                  p2_temp = PI1_p
                  Do While(.TRUE.)
                      IF(p1_temp.le.p .and. p.le.p2_temp)then
                          EXIT
                      ELSE
                          N_temp = N_temp + 1
                          p1_temp = p2_temp
                          p2_temp = p2_temp + p_mt1_mt2
                          t2 = t1
                          t1 = N_temp-t2
                      ENDIF
                  ENDDO
              ENDIF
          Endif   
!========================================================================
          Delta=r_ini*r_ini+a_spin*a_spin-two*r_ini+e*e 
          c_phi = a_spin*(r_ini*(two-e*ep)-(e*e+lambda*a_spin))/Delta
          c_time = ( (two+e*ep)*r_ini**three-e*e*r_ini*r_ini+(two*a_spin*(a_spin-lambda)&
                   +a_spin*a_spin*e*ep)*r_ini-e*e*a_spin*(a_spin-lambda) )/Delta
          index_p5(1)=-1
          index_p5(2)=-2
          index_p5(3)=0
          index_p5(4)=-4
          index_p5(5)=0
          !*****pp part***************************************
          If(lambda.ne.zero)then 
              cases_int=2
              call weierstrass_int_J3(tobs,tmu,dd,del,-a_add,b4,index_p5,abs(pp),integ5,cases_int)
              call weierstrass_int_J3(tobs,tmu,dd,del,-a_m,b4,index_p5,abs(pp),integ15,cases_int)
              pp_phi=(pp/(one-mu_tp1**two)+(integ5(2)*c_add-integ15(2)*c_m)/two)*lambda+c_phi*pp 
          else 
              pp_phi=c_phi*pp   
          endif
          cases_int=4
          call weierstrass_int_J3(tobs,tmu,dd,del,h,b4,index_p5,abs(pp),integ,cases_int)
          pp_sig=(pp*mu_tp1**two+integ(2)*mu_tp1*b0/two+integ(4)*b0**two/&
                  sixteen)*a_spin*a_spin+r_ini*r_ini*pp
          pp_t=pp_sig+c_time*pp  
          !*****p1 part***************************************
          If(t1.eq.0)then 
              p1_phi=zero
              p1_sig=zero 
              p1_t=zero
          else  
              If(lambda.ne.zero)then  
                  IF(PI1_phi .EQ. zero)THEN
                      cases_int=2 
                      call weierstrass_int_J3(tobs,infinity,dd,del,-a_add,b4,index_p5,PI0,integ5,cases_int)
                      call weierstrass_int_J3(tobs,infinity,dd,del,-a_m,b4,index_p5,PI0,integ15,cases_int)
                      PI1_phi=(PI0/(one-mu_tp1**two)+(integ5(2)*c_add-integ15(2)*c_m)/two)*lambda+c_phi*PI0
                  ENDIF 
                  p1_phi=PI1_phi-pp_phi 
              else 
                  IF(PI1_phi.eq.zero)PI1_phi=c_phi*PI0
                  p1_phi=PI1_phi-pp_phi     
              endif 
              IF(PI1_time .EQ. zero .or. PI1_sig.eq.zero)THEN  
                  cases_int=4  
                  call weierstrass_int_J3(tobs,infinity,dd,del,h,b4,index_p5,PI0,integ,cases_int)
                  PI1_sig=(PI0*mu_tp1**two+integ(2)*mu_tp1*b0/two+integ(4)*b0**two/&
                           sixteen)*a_spin*a_spin+r_ini*r_ini*PI0
                  PI1_time=PI1_sig+c_time*PI0  
              ENDIF
              p1_sig=PI1_sig-pp_sig
              p1_t=PI1_time-pp_t 
          endif 
          !*****p2 part***************************************
          If(t2.eq.0)then
              p2_phi=zero
              p2_t=zero
          else
              IF(lambda.ne.zero)then  
                  IF(PI2_phi .EQ. zero)THEN  
                      cases_int=2 
                      call weierstrass_int_J3(tp2,tobs,dd,del,-a_add,b4,index_p5,PI2_p,integ5,cases_int)
                      call weierstrass_int_J3(tp2,tobs,dd,del,-a_m,b4,index_p5,PI2_p,integ15,cases_int)
                      PI2_phi=(PI2_p/(one-mu_tp1*mu_tp1)+(integ5(2)*c_add-integ15(2)*c_m)/two)*lambda+c_phi*PI2_p
                  ENDIF
                  p2_phi=PI2_phi+pp_phi
              ELSE
                  IF(PI2_phi.eq.zero)PI2_phi=c_phi*PI2_p  
                  p2_phi=PI2_phi+pp_phi             
              ENDIF 
                
              IF(PI2_time .EQ. zero)THEN  
                  cases_int=4  
                  call weierstrass_int_J3(tp2,tobs,dd,del,h,b4,index_p5,PI2_p,integ,cases_int)
                  PI2_sig=(PI2_p*mu_tp1**two+integ(2)*mu_tp1*b0/two+integ(4)*b0**two/&
                                  sixteen)*a_spin*a_spin+r_ini*r_ini*PI2_p
                  PI2_time=PI2_sig+c_time*PI2_p  
              ENDIF 
              p2_sig=PI2_sig+pp_sig
              p2_t=PI2_time+pp_t   
          endif   
          !write(*,*)pp_phi,p1_phi,p2_phi,t1,t2
         !**************************************************************
          If(mobseqmtp)then 
              If(cos_ini .eq. mu_tp1)then
                  phyt= -pp_phi+two*(t1*p1_phi+t2*p2_phi)
                  timet= -pp_t+two*(t1*p1_t+t2*p2_t)
                  sigmat= -pp_sig+two*(t1*p1_sig+t2*p2_sig) 
              else
                  phyt=pp_phi+two*(t1*p1_phi+t2*p2_phi)
                  timet=pp_t+two*(t1*p1_t+t2*p2_t)
                  sigmat=pp_sig+two*(t1*p1_sig+t2*p2_sig)  
              endif
          else
              If(kt.lt.zero)then
                  phyt=pp_phi+two*(t1*p1_phi+t2*p2_phi) 
                  timet=pp_t+two*(t1*p1_t+t2*p2_t)
                  sigmat=pp_sig+two*(t1*p1_sig+t2*p2_sig)
              endif
              If(kt.gt.zero)then  
                  phyt=-pp_phi+two*(t1*p1_phi+t2*p2_phi)
                  timet=-pp_t+two*(t1*p1_t+t2*p2_t)
                  sigmat=-pp_sig+two*(t1*p1_sig+t2*p2_sig) 
              endif
          endif
          IF(mu_tp1.eq.one)phyt = phyt+(t1+t2)*PI
          !phyt = mod(phyt,twopi)
          !If(phyt .lt. zero)phyt=phyt+twopi
          count_num=count_num+1
      ELSE  
          If(kp_1.eq.kp.and.kt_1.eq.kt.and.lambda_1.eq.lambda.and.q_1.eq.q.and.&
          mve_1.eq.mve.and.ep_1.eq.ep.and.sin_ini_1.eq.sin_ini.and.e_1.eq.e&
          .and.cos_ini_1.eq.cos_ini.and.a_spin_1.eq.a_spin.and.r_ini_1.eq.r_ini&
          .and.theta_star_1.eq.theta_star)then   
 !***************************************************************************
          t1=0
          t2=0  
        !***********************************************************************    
          If(mu_tp1.eq.zero)then
              !photons are confined in the equatorial plane, so the integrations about \theta are valished. 
              mucos=zero
              phyt=p
              timet=phyt/Omega 
              sigmat=c_tau*timet 
              return
          endif
          !**************************************************************************
          If(a_spin.eq.zero .OR. mve.eq.one)then 
              CALL circ_mb(kp,kt,lambda,q,mve,ep,p,sin_ini,cos_ini,a_spin,e,&
                           r_ini,theta_star,phyt,timet,sigmat,mucos,t1,t2) 
              return
          endif   
          tmu=weierstrassP(p+PI01,g2,g3,dd,del)
          mucos = mu_tp1+b0/(four*tmu-b1)  
          !to get number of turn points of t1 and t2.
          !111111111*****************************************************************************************
          !mu=mu_tp+b0/(four*tmu-b1) 
          index_p5(1)=0
          cases_int=1
          call weierstrass_int_J3(tobs,tmu,dd,del,a4,b4,index_p5,rff_p,integ5,cases_int)  
          pp=integ5(1)
          p1=PI0-pp
          p2=p_mt1_mt2-p1  
!========================================================================
    !======== To determine the number of time_0 N_t1, N_t2 that the particle meets the
    !======== two turn points mu_tp1, mu_tp2 
          If(mobseqmtp)then
              p1_temp = zero
              p2_temp = p_mt1_mt2
              t1 = 0
              t2 = 0
              N_temp = 0
              If(cos_ini.eq.mu_tp1)then
                  Do While(.TRUE.)
                      IF(p1_temp.le.p .and. p.le.p2_temp)then
                          EXIT
                      ELSE
                          N_temp = N_temp + 1
                          p1_temp = p2_temp
                          p2_temp = p2_temp + p_mt1_mt2
                          t1 = t2
                          t2 = N_temp-t1
                      ENDIF
                  ENDDO
              Else If(cos_ini.eq.mu_tp2)then
                  Do While(.TRUE.)
                      IF(p1_temp.le.p .and. p.le.p2_temp)then
                          EXIT
                      ELSE
                          N_temp = N_temp + 1
                          p1_temp = p2_temp
                          p2_temp = p2_temp + p_mt1_mt2
                          t2 = t1
                          t1 = N_temp-t2
                      ENDIF
                  ENDDO
              Endif 
          Else
              p1_temp = zero
              t1 = 0
              t2 = 0
              N_temp = 0
              If(kt.gt.zero)then
                  p2_temp = PI2_p
                  Do While(.TRUE.)
                      IF(p1_temp.le.p .and. p.le.p2_temp)then
                          EXIT
                      ELSE
                          N_temp = N_temp + 1
                          p1_temp = p2_temp
                          p2_temp = p2_temp + p_mt1_mt2
                          t1 = t2
                          t2 = N_temp-t1
                      ENDIF
                  ENDDO
              ENDIF
              If(kt.lt.zero)then
                  p2_temp = PI1_p
                  Do While(.TRUE.)
                      IF(p1_temp.le.p .and. p.le.p2_temp)then
                          EXIT
                      ELSE
                          N_temp = N_temp + 1
                          p1_temp = p2_temp
                          p2_temp = p2_temp + p_mt1_mt2
                          t2 = t1
                          t1 = N_temp-t2
                      ENDIF
                  ENDDO
              ENDIF
          Endif   
!========================================================================
          index_p5(1)=-1
          index_p5(2)=-2
          index_p5(3)=0
          index_p5(4)=-4
          index_p5(5)=0
          !*****pp part***************************************
          If(lambda.ne.zero)then 
              cases_int=2
              call weierstrass_int_J3(tobs,tmu,dd,del,-a_add,b4,index_p5,abs(pp),integ5,cases_int)
              call weierstrass_int_J3(tobs,tmu,dd,del,-a_m,b4,index_p5,abs(pp),integ15,cases_int)
              pp_phi=(pp/(one-mu_tp1**two)+(integ5(2)*c_add-integ15(2)*c_m)/two)*lambda+c_phi*pp 
          else 
              pp_phi=c_phi*pp  
          endif 
          cases_int=4
          call weierstrass_int_J3(tobs,tmu,dd,del,h,b4,index_p5,abs(pp),integ,cases_int)
          pp_sig=(pp*mu_tp1**two+integ(2)*mu_tp1*b0/two+integ(4)*b0**two/&
                    sixteen)*a_spin*a_spin+r_ini*r_ini*pp
          pp_t=pp_sig+c_time*pp
          !*****p1 part***************************************
          If(t1.eq.0)then 
              p1_phi=zero
              p1_t=zero
          else  
              If(lambda.ne.zero)then  
                  IF(PI1_phi .EQ. zero)THEN
                      cases_int=2 
                      call weierstrass_int_J3(tobs,infinity,dd,del,-a_add,b4,index_p5,PI0,integ5,cases_int)
                      call weierstrass_int_J3(tobs,infinity,dd,del,-a_m,b4,index_p5,PI0,integ15,cases_int)
                      PI1_phi=(PI0/(one-mu_tp1**two)+(integ5(2)*c_add-integ15(2)*c_m)/two)*lambda+c_phi*PI0
                  ENDIF 
                  p1_phi=PI1_phi-pp_phi 
              else 
                  IF(PI1_phi.eq.zero)PI1_phi=c_phi*PI0 
                  p1_phi=PI1_phi-pp_phi     
              endif 
              IF(PI1_time .EQ. zero)THEN  
                  cases_int=4  
                  call weierstrass_int_J3(tobs,infinity,dd,del,h,b4,index_p5,PI0,integ,cases_int)
                  PI1_sig=(PI0*mu_tp1**two+integ(2)*mu_tp1*b0/two+integ(4)*b0**two/&
                                  sixteen)*a_spin*a_spin+r_ini*r_ini*PI0
                  PI1_time=PI1_sig+c_time*PI0  
              ENDIF
              p1_sig=PI1_sig-pp_sig
              p1_t=PI1_time-pp_t 
          endif 
          !*****p2 part***************************************
          If(t2.eq.0)then
              p2_phi=zero
              p2_t=zero
          else
              IF(lambda.ne.zero)then  
                  IF(PI2_phi .EQ. zero)THEN  
                      cases_int=2 
                      call weierstrass_int_J3(tp2,tobs,dd,del,-a_add,b4,index_p5,PI2_p,integ5,cases_int)
                      call weierstrass_int_J3(tp2,tobs,dd,del,-a_m,b4,index_p5,PI2_p,integ15,cases_int)
                      PI2_phi=(PI2_p/(one-mu_tp1*mu_tp1)+(integ5(2)*c_add-integ15(2)*c_m)/two)*lambda+c_phi*PI2_p 
                  ENDIF
                  p2_phi=PI2_phi+pp_phi
              ELSE
                  IF(PI2_phi.eq.zero)PI2_phi=c_phi*PI2_p 
                  p2_phi=PI2_phi+pp_phi           
              ENDIF 
                
              IF(PI2_time .EQ. zero)THEN  
                  cases_int=4  
                  call weierstrass_int_J3(tp2,tobs,dd,del,h,b4,index_p5,PI2_p,integ,cases_int)
                  PI2_sig=(PI2_p*mu_tp1**two+integ(2)*mu_tp1*b0/two+integ(4)*b0**two/&
                                  sixteen)*a_spin*a_spin+r_ini*r_ini*PI2_p
                  PI2_time=PI2_sig+c_time*PI2_p   
              ENDIF 
              p2_sig=PI2_sig+pp_sig 
              p2_t=PI2_time+pp_t   
          endif   
          !write(*,*)'kkk=',pp_phi,p1_phi,p2_phi,t1,t2
 !**************************************************************
          If(mobseqmtp)then 
              If(cos_ini .eq. mu_tp1)then
                  phyt= -pp_phi+two*(t1*p1_phi+t2*p2_phi)
                  timet= -pp_t+two*(t1*p1_t+t2*p2_t)
                  sigmat= -pp_sig+two*(t1*p1_sig+t2*p2_sig) 
              else
                  phyt=pp_phi+two*(t1*p1_phi+t2*p2_phi)
                  timet=pp_t+two*(t1*p1_t+t2*p2_t)
                  sigmat=pp_sig+two*(t1*p1_sig+t2*p2_sig)  
              endif
          else
              If(kt.lt.zero)then
                  phyt=pp_phi+two*(t1*p1_phi+t2*p2_phi) 
                  timet=pp_t+two*(t1*p1_t+t2*p2_t)
                  sigmat=pp_sig+two*(t1*p1_sig+t2*p2_sig)
              endif
              If(kt.gt.zero)then  
                  phyt=-pp_phi+two*(t1*p1_phi+t2*p2_phi)
                  timet=-pp_t+two*(t1*p1_t+t2*p2_t)
                  sigmat=-pp_sig+two*(t1*p1_sig+t2*p2_sig) 
              endif
          endif 
          IF(mu_tp1.eq.one)phyt = phyt+(t1+t2)*PI
          !phyt = mod(phyt,twopi)
          !If(phyt .lt. zero)phyt=phyt+twopi
       !***************************************************** 
          else
              count_num=1
              goto 30   
          endif    
      ENDIF 
      RETURN
      END SUBROUTINE CIR_ORBIT_PTA


!********************************************************************************************
      SUBROUTINE circ_mb(kp,kt,lambda,q,mve,ep,p,sin_ini,cos_ini,a_spin,e,r_ini,&
                         theta_star,phyt,timet,sigmat,mucos,t1,t2) 
!******************************************************************************************** 
!*     PURPOSE:   This routine computs the four B_L coordiants r,\mu,\phi,t and proper
!*                time \sigma of the spherical motion when black hole spin a_spin is zero or 
!*                constant of motion mve = 1. 
!*     INPUTS:   p--------------the independent variable.
!*               kp-------------k_{\phi}, initial \phi component of four momentum of a particle
!*                              measured under an LNRF. See equation (95).
!*               kt-------------k_{\theta}, initial \theta component of four momentum of a particle
!*                              measured under an LNRF. See equation (94).
!*               lambda,q,mve,ep-------constants of motion, defined by lambda=L_z/E, q=Q/E^2,
!*                                     mve = \mu_m/E, ep=epsilon/varepsilon. And mve = 1. 
!*               sin_ini,cos_ini-------sin_ini=sin(\theta_{ini}), cos_ini=cos(\theta_{ini}), where 
!*                                     \theta_{ini} is the initial \theta coordinate of the particle.
!*               a_spin,e--------------the spin and the electric charge of the black hole, 
!*                                     and -1 =< a_spin^2+e^2 <= 1 must be satisfied. And a_spin = 1. 
!*               r_ini-----------------the initial radial coordinate of the particle, i.e., the
!*                                     radius of the spherical motion. 
!*               theta_star------------the maximum or minimum value of the \theta coordinate of the geodesic.
!*                                     which also the \theta turning point of the spherical motion.       
!*     OUTPUTS:  phyt-----------the \phi coordinat of the photon.
!*               timet----------the t coordinats of the photon.
!*               sigmat---------the affine parameter \sigma.
!*               mucos----------the \theta coordinates of the photon, and mucos=cos(\theta).  
!*               t1,t2----------number of time_0 of photon meets turning points \mu_tp1 and \mu_tp2
!*                              respectively.      
!*     ROUTINES CALLED: metricg, circ_mb, weierstrass_int_J3, weierstrassP
!*     ACCURACY:   Machine.    
!*     AUTHOR:     Yang & Wang (2013)  
!*     DATE WRITTEN:  5 Jan 2013
!*     REVISIONS: ****************************************** 
      use constants 
      IMPLICIT NONE
      DOUBLE PRECISION kp,kt,lambda,q,mve,ep,p,sin_ini,cos_ini,a_spin,e,r_ini,phyt,timet,mucos,&
             kp_1,kt_1,lambda_1,q_1,mve_1,sin_ini_1,cos_ini_1,a_spin_1,r_ini_1,sigmat,AA,BB,&
             mu_tp1,mu_tp2,PI1,Ptotal,pp,p1,p2,PI1_phi,PI2_phi,PI1_time,PI2_time,PI1_sigma,&
             PI2_sigma,pp_sigma,p1_sigma,p2_sigma,pp_time,p1_time,p2_time,pp_phi,p1_phi,&
             p2_phi,mu2p,Delta,c_time,c_phi,PI2,mu,ep_1,e_1,theta_star,theta_star_1,p1_temp,&
             p2_temp
      integer :: t1,t2,count_num=1,i,j,N_temp
      save :: kp_1,kt_1,lambda_1,q_1,mve_1,sin_ini_1,cos_ini_1,a_spin_1,r_ini_1,PI1_phi,PI2_phi,&
              PI1_time,PI2_time,PI1_sigma,PI2_sigma,Delta,c_time,c_phi,PI1,PI2,mobseqmtp,ep_1,&
              e_1,AA,BB,mu_tp1,mu_tp2,Ptotal,theta_star_1
      logical :: mobseqmtp 

30    continue
      IF(count_num .eq. 1)THEN
          kp_1 = kp
          kt_1 = kt
          lambda_1 = lambda
          q_1 =q
          mve_1 = mve
          ep_1 = ep
          sin_ini_1 = sin_ini
          cos_ini_1 = cos_ini
          a_spin_1 = a_spin
          e_1 = e
          r_ini_1 = r_ini
          theta_star_1 = theta_star
          t1=0
          t2=0
          mobseqmtp = .false.
 
          IF(q.gt.zero)THEN
              AA = dsqrt((q+lambda*lambda)/q)
              BB = dsqrt(q)
          !^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
              If(kt.gt.zero)then
                  mu=sin(asin(cos_ini*AA)-p*BB*AA)/AA 
              else
                  If(kt.eq.zero)then
                      mu=cos(p*AA*BB)*cos_ini
                  else         
                      mu=sin(asin(cos_ini*AA)+p*AA*BB)/AA 
                  endif 
              endif
              mucos = mu  
          !****************************************************
              If(kt.ne.zero)then
                  mu_tp1=sqrt(q/(lambda**two+q))
                  mu_tp2=-mu_tp1 
              else
                  mu_tp1=abs(cos_ini)
                  mu_tp2=-mu_tp1
                  mobseqmtp=.true.
              endif
              If(abs(cos_ini).eq.one)mobseqmtp=.true.
          !^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ 
              If(mu_tp1.eq.zero)then
              !photons are confined in the equatorial plane, 
              !so the integrations about !\theta are valished.
                  timet = zero
                  sigmat = zero
                  phyt = zero
                  return
              endif  
          !***************************************************
              PI1=(PI/two-asin(cos_ini/mu_tp1))*mu_tp1/BB 
              Ptotal=PI*mu_tp1/BB
              PI2=Ptotal-PI1 
              pp=(asin(mu/mu_tp1)-asin(cos_ini/mu_tp1))*mu_tp1/BB 
              p1=PI1-pp
              p2=Ptotal-p1 
              PI1_phi=zero
              PI2_phi=zero
              PI1_time=zero
              PI2_time=zero
              PI1_sigma=zero
              PI2_sigma=zero 
!========================================================================
    !======== To determine the number of time_0 N_t1, N_t2 that the particle meets the
    !======== two turn points mu_tp1, mu_tp2 
              If(mobseqmtp)then
                  p1_temp = zero
                  p2_temp = ptotal
                  t1 = 0
                  t2 = 0
                  N_temp = 0
                  If(cos_ini.eq.mu_tp1)then
                      Do While(.TRUE.)
                          IF(p1_temp.le.p .and. p.le.p2_temp)then
                              EXIT
                          ELSE
                              N_temp = N_temp + 1
                              p1_temp = p2_temp
                              p2_temp = p2_temp + ptotal
                              t1 = t2
                              t2 = N_temp-t1
                          ENDIF
                      ENDDO
                  Else If(cos_ini.eq.mu_tp2)then
                      Do While(.TRUE.)
                          IF(p1_temp.le.p .and. p.le.p2_temp)then
                              EXIT
                          ELSE
                              N_temp = N_temp + 1
                              p1_temp = p2_temp
                              p2_temp = p2_temp + ptotal
                              t2 = t1
                              t1 = N_temp-t2
                          ENDIF
                      ENDDO
                  Endif 
              Else
                  p1_temp = zero
                  t1 = 0
                  t2 = 0
                  N_temp = 0
                  If(kt.gt.zero)then
                      p2_temp = PI2
                      Do While(.TRUE.)
                          IF(p1_temp.le.p .and. p.le.p2_temp)then
                              EXIT
                          ELSE
                              N_temp = N_temp + 1
                              p1_temp = p2_temp
                              p2_temp = p2_temp + ptotal
                              t1 = t2
                              t2 = N_temp-t1
                          ENDIF
                      ENDDO
                  ENDIF
                  If(kt.lt.zero)then
                      p2_temp = PI1
                      Do While(.TRUE.)
                          IF(p1_temp.le.p .and. p.le.p2_temp)then
                              EXIT
                          ELSE
                              N_temp = N_temp + 1
                              p1_temp = p2_temp
                              p2_temp = p2_temp + ptotal
                              t2 = t1
                              t1 = N_temp-t2
                          ENDIF
                      ENDDO
                  ENDIF
              Endif   
!========================================================================
              !p0 part!
              Delta = r_ini*r_ini-two*r_ini+a_spin*a_spin+e*e 
              c_phi = a_spin*(r_ini*(two-e*ep)-(e*e+lambda*a_spin))/Delta
              c_time = ( (two+e*ep)*r_ini**three-e*e*r_ini*r_ini+(two*a_spin*(a_spin-lambda)&
                       +a_spin*a_spin*e*ep)*r_ini-e*e*a_spin*(a_spin-lambda) )/Delta 
              pp_sigma = a_spin*a_spin*mveone_int(cos_ini,mu,one/AA)/BB+r_ini*r_ini*pp
              pp_time = pp_sigma+c_time*pp
              pp_phi = lambda*schwatz_int(cos_ini,mu,AA)/BB+c_phi*pp
              !******p1 part***********************************
              IF(t1 .eq. 0)THEN
                  p1_sigma=zero
                  p1_time=zero
                  p1_phi=zero
              ELSE
                  IF(PI1_time .eq. zero .or. PI1_sigma .eq. zero)THEN
                      PI1_sigma = a_spin*a_spin*mveone_int(cos_ini,mu_tp1,one/AA)/BB+r_ini*r_ini*PI1
                      PI1_time = PI1_sigma+c_time*PI1 
                  ENDIF
                  IF(PI1_phi .eq. zero)THEN
                      PI1_phi = lambda*schwatz_int(cos_ini,mu_tp1,AA)/BB+c_phi*PI1
                  ENDIF
                  p1_sigma = PI1_sigma-pp_sigma
                  p1_time = PI1_time-pp_time
                  p1_phi = PI1_phi-pp_phi
              ENDIF  
              !******p2 part***********************************
              IF(t2 .eq. 0)THEN
                  p2_sigma=zero
                  p2_time=zero
                  p2_phi=zero
              ELSE
                  IF(PI2_time .eq. zero .or. PI2_sigma .eq. zero)THEN
                      PI2_sigma = a_spin*a_spin*mveone_int(mu_tp2,cos_ini,one/AA)/BB+r_ini*r_ini*PI2
                      PI2_time = PI2_sigma+c_time*PI2 
                  ENDIF
                  IF(PI2_phi .eq. zero)THEN
                      PI2_phi = lambda*schwatz_int(mu_tp2,cos_ini,AA)/BB+c_phi*PI2
                  ENDIF
                  p2_sigma = PI2_sigma+pp_sigma
                  p2_time = PI2_time+pp_time
                  p2_phi = PI2_phi+pp_phi
              ENDIF
              !**********************************************  
              If(mobseqmtp)then
                  If(cos_ini.eq.mu_tp1)then  
                      sigmat = -pp_sigma+two*(t1*p1_sigma+t2*p2_sigma)
                      timet = -pp_time+two*(t1*p1_time+t2*p2_time) 
                      phyt = -pp_phi+two*(t1*p1_phi+t2*p2_phi) 
                  else
                      sigmat = pp_sigma+two*(t1*p1_sigma+t2*p2_sigma)
                      timet = pp_time+two*(t1*p1_time+t2*p2_time) 
                      phyt = pp_phi+two*(t1*p1_phi+t2*p2_phi)  
                  endif  
              else
                 If(kt.lt.zero)then
                      sigmat = pp_sigma+two*(t1*p1_sigma+t2*p2_sigma)
                      timet = pp_time+two*(t1*p1_time+t2*p2_time) 
                      phyt = pp_phi+two*(t1*p1_phi+t2*p2_phi) 
                 endif  
                 If(kt.gt.zero)then
                      sigmat = -pp_sigma+two*(t1*p1_sigma+t2*p2_sigma)
                      timet = -pp_time+two*(t1*p1_time+t2*p2_time) 
                      phyt = -pp_phi+two*(t1*p1_phi+t2*p2_phi) 
                 endif   
              endif
              If(theta_star.eq.zero.or.theta_star.eq.180.D0)phyt = phyt+(t1+t2)*PI
          ELSE
              !write(unit=6,fmt=*)'phyt_schwatz(): q<0, which is a affending',&
              !  'value, the program should be',&  
              !  'stoped! and q = ',q
              !stop
              mucos=cos_ini 
              t1 = 0
              t2 = 0
              phyt = zero 
              timet = zero
              sigmat = zero
          ENDIF
      ELSE 
          IF(kp_1 .eq. kp.and.kt_1 .eq. kt.and.lambda_1 .eq. lambda.and.q_1 .eq.q.and.&
          mve_1 .eq. mve.and.sin_ini_1 .eq. sin_ini.and.cos_ini_1 .eq. cos_ini.and.a_spin_1 &
          .eq. a_spin.and.r_ini_1 .eq. r_ini .and.ep_1.eq.ep.and.e_1.eq.e.and.&
          theta_star_1.eq.theta_star)THEN
      !*******************************************************************************
            IF(q.gt.zero)THEN 
          !^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
              If(kt.gt.zero)then
                  mu=sin(asin(cos_ini*AA)-p*BB*AA)/AA 
              else
                  If(kt.eq.zero)then
                      mu=cos(p*AA*BB)*cos_ini
                  else         
                      mu=sin(asin(cos_ini*AA)+p*AA*BB)/AA 
                  endif 
              endif
              mucos = mu   
          !^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ 
              If(mu_tp1.eq.zero)then
              !photons are confined in the equatorial plane, 
              !so the integrations about !\theta are valished.
                  timet = zero
                  sigmat = zero
                  phyt = zero
                  return
              endif  
          !*************************************************** 
              pp=(asin(mu/mu_tp1)-asin(cos_ini/mu_tp1))*mu_tp1/BB 
              p1=PI1-pp
              p2=Ptotal-p1  
!========================================================================
    !======== To determine the number of time_0 N_t1, N_t2 that the particle meets the
    !======== two turn points mu_tp1, mu_tp2 
              If(mobseqmtp)then
                  p1_temp = zero
                  p2_temp = ptotal
                  t1 = 0
                  t2 = 0
                  N_temp = 0
                  If(cos_ini.eq.mu_tp1)then
                      Do While(.TRUE.)
                          IF(p1_temp.le.p .and. p.le.p2_temp)then
                              EXIT
                          ELSE
                              N_temp = N_temp + 1
                              p1_temp = p2_temp
                              p2_temp = p2_temp + ptotal
                              t1 = t2
                              t2 = N_temp-t1
                          ENDIF
                      ENDDO
                  Else If(cos_ini.eq.mu_tp2)then
                      Do While(.TRUE.)
                          IF(p1_temp.le.p .and. p.le.p2_temp)then
                              EXIT
                          ELSE
                              N_temp = N_temp + 1
                              p1_temp = p2_temp
                              p2_temp = p2_temp + ptotal
                              t2 = t1
                              t1 = N_temp-t2
                          ENDIF
                      ENDDO
                  Endif 
              Else
                  p1_temp = zero
                  t1 = 0
                  t2 = 0
                  N_temp = 0
                  If(kt.gt.zero)then
                      p2_temp = PI2
                      Do While(.TRUE.)
                          IF(p1_temp.le.p .and. p.le.p2_temp)then
                              EXIT
                          ELSE
                              N_temp = N_temp + 1
                              p1_temp = p2_temp
                              p2_temp = p2_temp + ptotal
                              t1 = t2
                              t2 = N_temp-t1
                          ENDIF
                      ENDDO
                  ENDIF
                  If(kt.lt.zero)then
                      p2_temp = PI1
                      Do While(.TRUE.)
                          IF(p1_temp.le.p .and. p.le.p2_temp)then
                              EXIT
                          ELSE
                              N_temp = N_temp + 1
                              p1_temp = p2_temp
                              p2_temp = p2_temp + ptotal
                              t2 = t1
                              t1 = N_temp-t2
                          ENDIF
                      ENDDO
                  ENDIF
              Endif   
!========================================================================
              !p0 part!   
              pp_sigma = a_spin*a_spin*mveone_int(cos_ini,mu,one/AA)/BB+r_ini*r_ini*pp
              pp_time = pp_sigma+c_time*pp
              pp_phi = lambda*schwatz_int(cos_ini,mu,AA)/BB+c_phi*pp
              !******p1 part***********************************
              IF(t1 .eq. 0)THEN
                  p1_sigma=zero
                  p1_time=zero
                  p1_phi=zero
              ELSE
                  IF(PI1_time .eq. zero .or. PI1_sigma .eq. zero)THEN
                      PI1_sigma = a_spin*a_spin*mveone_int(cos_ini,mu_tp1,one/AA)/BB+r_ini*r_ini*PI1
                      PI1_time = PI1_sigma+c_time*PI1 
                  ENDIF
                  IF(PI1_phi .eq. zero)THEN
                      PI1_phi = lambda*schwatz_int(cos_ini,mu_tp1,AA)/BB+c_phi*PI1
                  ENDIF
                  p1_sigma = PI1_sigma-pp_sigma
                  p1_time = PI1_time-pp_time
                  p1_phi = PI1_phi-pp_phi
              ENDIF  
              !******p2 part***********************************
              IF(t2 .eq. 0)THEN
                  p2_sigma=zero
                  p2_time=zero
                  p2_phi=zero
              ELSE
                  IF(PI2_time .eq. zero .or. PI2_sigma .eq. zero)THEN
                      PI2_sigma = a_spin*a_spin*mveone_int(mu_tp2,cos_ini,one/AA)/BB+r_ini*r_ini*PI2
                      PI2_time = PI2_sigma+c_time*PI2 
                  ENDIF
                  IF(PI2_phi .eq. zero)THEN
                      PI2_phi = lambda*schwatz_int(mu_tp2,cos_ini,AA)/BB+c_phi*PI2
                  ENDIF
                  p2_sigma = PI2_sigma+pp_sigma
                  p2_time = PI2_time+pp_time
                  p2_phi = PI2_phi+pp_phi
              ENDIF
              !**********************************************  
              If(mobseqmtp)then
                  If(cos_ini.eq.mu_tp1)then  
                      sigmat = -pp_sigma+two*(t1*p1_sigma+t2*p2_sigma)
                      timet = -pp_time+two*(t1*p1_time+t2*p2_time) 
                      phyt = -pp_phi+two*(t1*p1_phi+t2*p2_phi) 
                  else
                      sigmat = pp_sigma+two*(t1*p1_sigma+t2*p2_sigma)
                      timet = pp_time+two*(t1*p1_time+t2*p2_time) 
                      phyt = pp_phi+two*(t1*p1_phi+t2*p2_phi)  
                  endif  
              else
                 If(kt.lt.zero)then
                      sigmat = pp_sigma+two*(t1*p1_sigma+t2*p2_sigma)
                      timet = pp_time+two*(t1*p1_time+t2*p2_time) 
                      phyt = pp_phi+two*(t1*p1_phi+t2*p2_phi) 
                 endif  
                 If(kt.gt.zero)then
                      sigmat = -pp_sigma+two*(t1*p1_sigma+t2*p2_sigma)
                      timet = -pp_time+two*(t1*p1_time+t2*p2_time) 
                      phyt = -pp_phi+two*(t1*p1_phi+t2*p2_phi) 
                 endif   
              endif
              If(theta_star.eq.zero.or.theta_star.eq.180.D0)phyt = phyt+(t1+t2)*PI
            ELSE
              !write(unit=6,fmt=*)'phyt_schwatz(): q<0, which is a affending',&
              !  'value, the program should be',&  
              !  'stoped! and q = ',q
              !stop
              mucos=cos_ini 
              t1 = 0
              t2 = 0
              phyt = zero 
              timet = zero
              sigmat = zero
            ENDIF
          ELSE
              count_num = 1
              goto 30
          ENDIF
      ENDIF
      RETURN
      END SUBROUTINE circ_mb


!*****************************************************************************************************
      subroutine metricg(r_ini,sin_ini,cos_ini,a_spin,e,somiga,expnu,exppsi,expmu1,expmu2)
!*****************************************************************************************************
!*     PURPOSE:  Computes Kerr_Newmann metric, exp^\nu, exp^\psi, exp^mu1, exp^\mu2, and omiga at position:
!*               r_ini, \theta_{ini}.     
!*     INPUTS:   r_ini-----------------the initial radial coordinate of the particle. 
!*               sin_ini,cos_ini-------sin_ini=sin(\theta_{ini}), cos_ini=cos(\theta_{ini}), where 
!*                                     \theta_{ini} is the initial \theta coordinate of the particle.
!*               a_spin,e--------------the spin and the electric charge of the black hole, 
!*                                     and -1 =< a_spin^2+e^2 <= 1 must be satisfied.        
!*     OUTPUTS:  somiga,expnu,exppsi,expmu1,expmu2------------Kerr_Newmann metrics under Boyer-Lindquist coordinates.
!*     ROUTINES CALLED: NONE
!*     ACCURACY:   Machine.    
!*     AUTHOR:     Yang & Wang (2012).  
!*     DATE WRITTEN:  5 Jan 2012.
!*     REVISIONS: ****************************************** 
      implicit none
      Double precision r_ini,theta,a_spin,e,Delta,bigA,two,sin_ini,cos_ini,one,sigma
      Double precision somiga,expnu,exppsi,expmu1,expmu2      

      two=2.D0      
      one=1.D0
      Delta=r_ini**two-two*r_ini+a_spin**two+e*e
      sigma=r_ini**two+(a_spin*cos_ini)**two
      bigA=(r_ini**two+a_spin**two)**two-(a_spin*sin_ini)**two*Delta
      somiga=(two*r_ini-e*e)*a_spin/bigA
      expnu=sqrt(sigma*Delta/bigA)
      exppsi=sin_ini*sqrt(bigA/sigma)
      expmu1=sqrt(sigma/Delta)
      expmu2=sqrt(sigma)
      return      
      End subroutine metricg      

 
!******************************************************************************************** 
      Subroutine lambdaq(alpha,beta,velt,r_ini,signcharge,sin_ini,&
                        cos_ini,a_spin,e,scal,velocity,lambda,q,mve,ep,k) 
!********************************************************************************************
!*     PURPOSE:  Computes constants of motion from impact parameters alpha and beta by using 
!*               formulae (86) and (87) in Yang & Wang (2013).    
!*     INPUTS:   alpha,beta-----Impact parameters.
!*               r_ini-----------radial coordinate of observer or the initial position of photon.
!*               sin_ini,cos_ini-------sin_ini=sin(\theta_{ini}), cos_ini=cos(\theta_{ini}), where 
!*                                     \theta_{ini} is the initial \theta coordinate of the particle.
!*               a_spin---------spin of black hole, on interval [-1,1].  
!*               scal-----------a dimentionless parameter to control the size of the images.
!*                              Which is usually be set to 1.D0. 
!*               velocity(1:3)--Array of physical velocities of the observer or emitter with respect to
!*                              LNRF.        
!*     OUTPUTS:  k(1:4)---------array of p_r, p_theta, p_phi, p_t, which are the components of 
!*                              four momentum of a photon measured under the LNRF frame, and 
!*                              defined by equations (82)-(85) in Yang & Wang (2012). 
!*               lambda,q,mve,k-------motion constants, defined by lambda=L_z/E, q=Q/E^2.            
!*     ROUTINES CALLED: NONE.
!*     ACCURACY:   Machine.    
!*     AUTHOR:     Yang & Wang (2012).  
!*     DATE WRITTEN:  5 Jan 2012.
!*     REVISIONS: ****************************************** 
      implicit none
      Double precision f1234(4),r_ini,sin_ini,cos_ini,a_spin,lambda,q,As,Bs,Er,A1,a2,b1,b2,c1,c2,d1,d2,&
             Delta,zero,one,two,four,gff,Sigma,Ac,Bc,Cc,at,Bt,lambdap,lambdam,qp,qm,&
             RaB2,scal,Vr,Vt,Vp,gama,gama_tp,gama_p,F1,F2,NN1,NN2,DD1,DD2,Aab,expnu2,&
             eppsi2,epnu2,epmu12,epmu22,bigA,G1,G2,KF,three,alpha,beta,e,signcharge,&
             velocity(3),lambda2,q1,q2,q3,velt,vrp,vtp,vpp,gamap,k(4),mve,ep,& 
             somega,epm,epp 
      parameter(zero=0.D0,one=1.D0,two=2.D0,three=3.D0,four=4.D0)

      if(abs(beta).lt.1.D-7)beta=zero
      if(abs(alpha).lt.1.D-7)alpha=zero
      at=alpha/scal/r_ini
      Bt=beta/scal/r_ini      
      
      vrp = velt/dsqrt(one+at*at+Bt*Bt)  
      vtp = Bt*vrp
      vpp = at*vrp
      gamap = one/dsqrt(one-vrp*vrp-vtp*vtp-vpp*vpp)
 
      Vr=velocity(1)
      Vt=velocity(2)
      Vp=velocity(3) 
      gama=one/dsqrt(one-(Vr**two+Vt**two+Vp**two)) 
      If(vrp*vrp+vtp*vtp+vpp*vpp.gt.one .or. Vr*Vr+Vt*Vt+Vp*Vp.gt.one)then
          write(*,*)vrp*vrp+vtp*vtp+vpp*vpp,Vr*Vr+Vt*Vt+Vp*Vp
          write(*,*)'lambdaq(): super the speed of light.'
         ! stop      
      endif  

      k(4) = gama*(one+Vr*vrp+Vt*vtp+Vp*vpp)
      k(1) = gama*Vr+vrp*(one+gama*gama*Vr*Vr/(one+gama))+gama*gama*Vr*Vt*vtp/(one+gama)+&
                     gama*gama*Vr*Vp*vpp/(one+gama)
      k(2) = gama*Vt+gama*gama*Vr*Vt*vrp/(one+gama)+vtp*(one+gama*gama*Vt*Vt/(one+gama))+&
                     gama*gama*Vt*Vp*vpp/(one+gama)
      k(3) = gama*Vp+gama*gama*Vp*Vr*vrp/(one+gama)+gama*gama*Vp*Vt*vtp/(one+gama)+&
                     vpp*(one+gama*gama*Vp*Vp/(one+gama))
      k(1) = -k(1)  
      If(dabs(k(1)).lt.1.D-7)k(1)=zero
      If(dabs(k(2)).lt.1.D-7)k(2)=zero
      If(dabs(k(3)).lt.1.D-7)k(3)=zero
      If(dabs(k(4)).lt.1.D-7)k(4)=zero

      Delta=r_ini**two-two*r_ini+a_spin**two+e*e
      Sigma=r_ini**two+(a_spin*cos_ini)**two
      bigA=(r_ini**two+a_spin**two)**two-(a_spin*sin_ini)**two*Delta

      somega=(two*r_ini-e*e)*a_spin/bigA
      expnu2=Sigma*Delta/bigA
      eppsi2=sin_ini**two*bigA/Sigma
      epmu12=Sigma/Delta
      epmu22=Sigma

      A1 = k(3)/(dsqrt(Delta)*Sigma/bigA*k(4)+k(3)*somega*sin_ini) 
      lambda = A1*sin_ini 

      mve = (one-lambda*somega)/dsqrt(expnu2)/gamap/k(4)
      q = (A1*A1-a_spin*a_spin*(one-mve*mve))*cos_ini*cos_ini+mve*mve*Sigma*gamap*gamap*k(2)*k(2)
 
      IF(e.ne.zero)THEN
          a2 = e*e*r_ini*r_ini
          b2 = two*e*r_ini*(r_ini*r_ini+a_spin*a_spin-a_spin*lambda) 
          c2 = r_ini**four*(one-mve*mve)+two*mve*mve*r_ini**three-(q+lambda*lambda+e*e*mve*mve+&
              a_spin*a_spin*(mve*mve-one))*r_ini**two+two*(q+(a_spin-lambda)*(a_spin-lambda))*r_ini-&
              e*e*(a_spin-lambda)*(a_spin-lambda)-(a_spin*a_spin+e*e)*q-&
              Sigma*Delta*(gamap*gama*mve*k(1))**two 
          epp = (-b2+dsqrt(b2*b2-four*a2*c2))/two/a2
          epm = (-b2-dsqrt(b2*b2-four*a2*c2))/two/a2
          IF(signcharge.gt.zero)THEN
              ep = epp
          ELSE
              ep = epm
          ENDIF
          !write(*,*)'ep=',epp,epm,ep,a2,b2,c2
      ELSE
          ep = zero  
          !write(*,*)'ep=',ep 
      ENDIF 
   
      return
      End subroutine lambdaq


!********************************************************************************************************
      Subroutine lambdaqm(vptl,sin_ini,cos_ini,a_spin,e,r_ini,signcharge,vobs,lambda,q,mve,ep,kvec) 
!********************************************************************************************************
!*     PURPOSE:  Computes constants of motion from initial conditions: the physical velocities vptl(1:3) of the 
!*               particle with respect to the assumed emitter, and the physical velocities of the assumed 
!*               emitter with respect to the LNRF reference. See equations (97)-(104) in Yang & Wang (2013), A&A.
!*               And equations (92)-(96).    
!*     INPUTS:   vptl(1:3)------Array contains the the physical velocities of the particle with respect 
!*                              to the reference of an assumed emitter. 
!*               sin_ini,cos_ini-------sin_ini=sin(\theta_{ini}), cos_ini=cos(\theta_{ini}), where 
!*                                     \theta_{ini} is the initial \theta coordinate of the particle.
!*               a_spin,e--------------the spin and the electric charge of the black hole, 
!*                                     and -1 =< a_spin^2+e^2 <= 1 must be satisfied.  
!*               r_ini-----------------the initial radial coordinate of the particle. 
!*               signcharge------------To specify the sign of the electric charge of the particle.
!*                                     If signcharge>0, then ep>0
!*                                     If signcharge<0, then ep<0.
!*               vobs(1:3)-------------Array of physical velocities of the observer or emitter with respect to
!*                                     the LNRF reference.        
!*     OUTPUTS:  kvec(1:4)-------------array of p_r, p_theta, p_phi, p_t, which are the components of 
!*                                     four momentum of a particle measured under the LNRF frame, and 
!*                                     defined by equations (92)-(95) in Yang & Wang (2012). 
!*               lambda,q,mve,ep-------constants of motion, defined by lambda=L_z/E, q=Q/E^2,
!*                                     mve = \mu_m/E, ep=epsilon/varepsilon.            
!*     ROUTINES CALLED: NONE.
!*     ACCURACY:   Machine.    
!*     AUTHOR:     Yang & Wang (2012).  
!*     DATE WRITTEN:  5 Jan 2012.
!*     REVISIONS: ******************************************************************************************* 
      implicit none
      Double precision vptl(3),lambda,q,sin_ini,cos_ini,a_spin,r_ini,zero,one,two,&
             three,four,gamap,mve,mve1,A1,e,ep,epp,epm,&
             pt,pp,cost,sint,cosp,sinp,Ac,Bc,Cc,moment(4),vobs(3),Vr,Vt,Vp,&
             b1,c1,d1,a2,b2,c2,d2,gama,gama_tp,gama_p,kvec(4),KF,F1,F2,&
             Delta,Er,Sigma,bigA,eppsi2,epmu12,epmu22,NN1,NN2,DD1,p_E,somega,&
             expnu2,signcharge 
      parameter(zero=0.D0,one=1.D0,two=2.D0,three=3.D0,four=4.D0) 
      integer cases

      gamap = one/dsqrt(one-vptl(1)*vptl(1)-vptl(2)*vptl(2)-vptl(3)*vptl(3))
      gama = one/dsqrt(one-vobs(1)*vobs(1)-vobs(2)*vobs(2)-vobs(3)*vobs(3))
      gama_tp = one/dsqrt(one-vobs(2)*vobs(2)-vobs(3)*vobs(3))
      gama_p = one/dsqrt(one-vobs(3)*vobs(3))
      kvec(1)=(vobs(1)+vptl(1)/gama_tp)
      kvec(2)=(gama*vobs(2)+gama*gama_tp*vobs(1)*vobs(2)*vptl(1)+gama_tp/gama_p*vptl(2))

      kvec(3)=(gama*vobs(3)+gama*gama_tp*vobs(1)*vobs(3)*vptl(1)+gama_tp*gama_p*vobs(2)*&
                      vobs(3)*vptl(2)+gama_p*vptl(3)) 
      kvec(4)=(gama+gama*gama_tp*vobs(1)*vptl(1)+gama_tp*gama_p*vobs(2)*&
                     vptl(2)+gama_p*vobs(3)*vptl(3))
      If(abs(kvec(1)).lt.1.D-7)kvec(1)=zero
      If(abs(kvec(2)).lt.1.D-7)kvec(2)=zero
      If(abs(kvec(3)).lt.1.D-7)kvec(3)=zero
      If(abs(kvec(4)).lt.1.D-7)kvec(4)=zero

      Delta=r_ini**two-two*r_ini+a_spin**two+e*e
      Sigma=r_ini**two+(a_spin*cos_ini)**two
      bigA=(r_ini**two+a_spin**two)**two-(a_spin*sin_ini)**two*Delta

      somega=(two*r_ini-e*e)*a_spin/bigA
      expnu2=Sigma*Delta/bigA
      eppsi2=sin_ini**two*bigA/Sigma
      epmu12=Sigma/Delta
      epmu22=Sigma
      Er=r_ini**four+(a_spin*r_ini)**two+two*a_spin**two*r_ini
      
      A1=kvec(3)/(kvec(4)*dsqrt(Delta)*Sigma/bigA+kvec(3)*somega*sin_ini) 
      lambda = A1*sin_ini 
      mve = (one-lambda*somega)/dsqrt(expnu2)/gamap/kvec(4)
      q = (A1*A1-a_spin*a_spin*(one-mve*mve))*cos_ini*cos_ini+mve*mve*Sigma*gamap*gamap*kvec(2)*kvec(2)
 
      IF(e.ne.zero)THEN
          a2 = e*e*r_ini*r_ini
          b2 = two*e*r_ini*(r_ini*r_ini+a_spin*a_spin-a_spin*lambda) 
          c2 = r_ini**four*(one-mve*mve)+two*mve*mve*r_ini**three-(q+lambda*lambda+e*e*mve*mve+&
               a_spin*a_spin*(mve*mve-one))*r_ini**two+two*(q+(a_spin-lambda)*(a_spin-lambda))*r_ini-&
               e*e*(a_spin-lambda)*(a_spin-lambda)-(a_spin*a_spin+e*e)*q-&
               Sigma*Delta*(gamap*gama*mve*kvec(1))**two!epmu12*(mve*Delta*gama*gamap*kvec(1))**two  
          epp = (-b2+dsqrt(b2*b2-four*a2*c2))/two/a2
          epm = (-b2-dsqrt(b2*b2-four*a2*c2))/two/a2
          IF(signcharge.gt.zero)THEN
              ep = epp
          ELSE
              ep = epm
          ENDIF
          !write(*,*)'ep=',epp,epm,ep,a2,b2,c2
      ELSE
          ep = zero  
            !write(*,*)'ep=',ep 
      ENDIF            
      return
      End subroutine lambdaqm


!*****************************************************************************************************
      Subroutine spherical_motion_constants(r,theta_star,a_spin,e,sign_charge,lambda,q,mve,ep)
!*****************************************************************************************************
!*     PURPOSE:  Computes constants of spherical motion from conditions: theta_star, a_spin, e.
!*               See equations (110)-(113) in Yang & Wang (2013), A&A.   
!*     INPUTS:   r--------------The radius of the spherical motion.
!*               theta_star-----The theta coordinate of the turning point in the spherical motion. 
!*               a_spin,e--------------the spin and the electric charge of the black hole, 
!*                                     and -1 =< a_spin^2+e^2 <= 1 must be satisfied.  
!*               r_ini-----------------the initial radial coordinate of the particle. 
!*               sign_charge-----------To specify the sign of the electric charge of the particle.
!*                                     If signcharge>0, then ep>0
!*                                     If signcharge<0, then ep<0.       
!*     OUTPUTS:  lambda,q,mve,ep-------constants of motion, defined by lambda=L_z/E, q=Q/E^2,
!*                                     mve = \mu_m/E, ep=epsilon/varepsilon.            
!*     ROUTINES CALLED: NONE.
!*     ACCURACY:   Machine.    
!*     AUTHOR:     Yang & Wang (2012).  
!*     DATE WRITTEN:  5 Jan 2012.
!*     REVISIONS: ******************************************************************************************* 
      use constants
      implicit none
      Double precision :: r,theta_star,a_spin,e,lambda,q,mve,ep,&
             Evsm,Qvsm,Lvsm,P,r2,a2,e2,Delta,Sigma,Denome,&
             a1,b1,c1,sign_charge,sin_star,cos_star,mu2,sin2,AQ
      
      If(theta_star.ne.90.D0)then
          sin_star=dsin(theta_star*dtor)
          cos_star=dcos(theta_star*dtor)
      else
          sin_star=one
          cos_star=zero
      endif
    
      mu2 = cos_star*cos_star
      sin2 = sin_star*sin_star
      a2 = a_spin*a_spin
      e2 = e*e
      r2 = r*r
      P = r2-a2*mu2-e2*r
      Delta = r2-two*r+a2+e2
      Sigma = r2+a2*mu2

      AQ = (r2+a2)**two*(r-e2)+a2*( (r*(r2-a2)+e2*a2)*sin2-(two*r-e2)**two*mu2 )

      Denome = dsqrt( Sigma*(-P+(Delta-a2*sin2)*r+two*a_spin*sin_star*dsqrt(r*P)) )
      Evsm = ( a_spin*sin_star*dsqrt(P)+(Delta-a2*sin2)*dsqrt(r) )/Denome
      Lvsm = sin_star*( (r2+a2)*dsqrt(P)-dsqrt(r)*a_spin*sin_star*(two*r-e2) )/Denome
      Qvsm = r*mu2*( AQ-two*a_spin*sin_star*(two*r-e2)*r*dsqrt(r*P) )/Denome/Denome
      lambda = Lvsm/Evsm
      q = Qvsm/Evsm/Evsm
      mve = one/Evsm
 
      If(e.ne.zero)then
          a1 = e*e*r*r
          b1 = two*e*r*(r*r+a_spin*a_spin-a_spin*lambda) 
          c1 = r**four*(one-mve*mve)+two*mve*mve*r**three-(q+lambda*lambda+e*e*mve*mve+&
                a_spin*a_spin*(mve*mve-one))*r**two+two*(q+(a_spin-lambda)*(a_spin-lambda))*r-&
                e*e*(a_spin-lambda)*(a_spin-lambda)-(a_spin*a_spin+e*e)*q 
          If(sign_charge.gt.zero)then
              ep = (-b1+dsqrt(b1*b1-four*a1*c1))/(two*a1) 
          else
              ep = (-b1-dsqrt(b1*b1-four*a1*c1))/(two*a1) 
          endif
      else
          ep = zero
      endif
 
      return         
      End Subroutine spherical_motion_constants 


!********************************************************************************************
      FUNCTION p_total(kvec,lambda,q,mve,ep,sin_ini,cos_ini,a_spin,e,r_ini) 
!******************************************************************************************** 
!*     PURPOSE:  Computes the integral value of \int^r dr (R)^{-1/2}, from the starting position to
!*               the termination----either at infinity or at the event horizon.   
!*     INPUTS:   kvec(4)---------------an array contains k_{r}, k_{\theta}, k_{\phi}, and k_{t}, which are 
!*                                     defined by equations (92)-(96). k_{i} can also be regarded as 
!*                                     components of four-momentum of a photon measured under the LNRF frame.   
!*               lambda,q,mve,ep-------constants of motion, defined by lambda=L_z/E, q=Q/E^2,
!*                                     mve = \mu_m/E, ep=epsilon/varepsilon. 
!*               sin_ini,cos_ini-------sin_ini=sin(\theta_{ini}), cos_ini=cos(\theta_{ini}), where 
!*                                     \theta_{ini} is the initial \theta coordinate of the particle.
!*               a_spin,e--------------the spin and the electric charge of the black hole, 
!*                                     and -1 =< a_spin^2+e^2 <= 1 must be satisfied. 
!*               r_ini-----------------the initial radial coordinate of the particle.    
!*     OUTPUTS:  p_total--------which is the value of integrals \int^r dr (R)^{-1/2}, along a 
!*                              whole geodesic, that is from the starting position to either go to
!*                              infinity or fall in to black hole.          
!*     ROUTINES CALLED: root3, weierstrass_int_J3, radiustp, weierstrassP, EllipticF, carlson_doublecomplex5 
!*     ACCURACY:   Machine.    
!*     AUTHOR:     Yang & Wang (2012)  
!*     DATE WRITTEN:  5 Jan 2012
!*     REVISIONS: ******************************************  
      USE constants
      IMPLICIT NONE
      DOUBLE PRECISION phyr,re,sin_ini,cos_ini,a_spin,rhorizon,q,lambda,integ,kvec(4),&
             bc,cc,dc,ec,b0,b1,b2,b3,g2,g3,tobs,tp,pp,p1,p2,PI0,p1I0,p1J1,p1J2,&
             u,v,w,L1,L2,thorizon,m2,pinf,sn,cn,dn,rp,rm,B_add,come,p_total,&
             y,x,f1,g1,h1,f2,h2,a5,b5,a4,b4,integ0,integ1,integ2,r_ini,ttp,mve,mve_1,&
             PI1,PI2,tinf,integ05(5),integ5(5),integ15(5),pp2,tp1,c_temp,k1,k2,&
             r_tp1,r_tp2,t_inf,tp2,kvecr,kvect,p_temp,PI0_obs_inf,PI0_total,PI0_obs_hori,&
             PI0_obs_tp2,PI01,timer,affr,r_coord,cr,dr,rff_p,p_t1_t2,s,ac1,bc1,cc1,&
             h,pp_time,pp_phi,pp_aff,p1_phi,p1_time,p1_aff,sqrtcome,&
             p2_phi,p2_time,p2_aff,time_temp,sqt3,p_tp1_tp2,PI2_p,PI1_p,alpha1,alpha2,&
             PI1_phi,PI2_phi,PI1_time,PI2_time,PI1_aff,PI2_aff,e,ep,e_1,ep_1,&
             Atp,Atm,Dtp,Dtm,trp,trm,App,Apm,Dpp,Dpm,ar(5),Btp,Btm,Bpp,Bpm  
      COMPLEX*16 bb(1:4),dd(3)
      INTEGER :: reals,i,j,t1,t2,p5,p4,index_p5(5),del,cases_int,cases 
      LOGICAL :: r_ini_eq_rtp,indrhorizon,r1eqr2    
 
          kvecr = kvec(1)
          kvect = kvec(2)
          rp=one+sqrt(one-a_spin**two-e*e) 
          rhorizon=rp 
 
          b4=one
          a4=zero 
          IF(mve .EQ. one)THEN
              p_total = p_total_MB(kvecr,kvect,lambda,q,mve,ep,sin_ini,cos_ini,a_spin,e,r_ini) 
              return
          ENDIF 
          r_ini_eq_rtp=.false.
          indrhorizon=.false.
          call radiustp(kvecr,a_spin,e,r_ini,lambda,q,mve,ep,r_tp1,r_tp2,&
                                     reals,r_ini_eq_rtp,indrhorizon,r1eqr2,cases,bb) 
          come = mve*mve-one    
          If(reals.ne.0)then  !** R(r)=0 has real roots and turning points exists in radial r.
              ar(1)=-come 
              ar(2)=two*(mve*mve+e*ep) 
              ar(3)=-(a_spin*a_spin*come+lambda*lambda+q+e*e*(mve*mve-ep*ep)) 
              ar(4)=two*(q+(lambda-a_spin)*(lambda-a_spin)+a_spin*e*ep*(a_spin-lambda)) 
              ar(5)=-q*(a_spin*a_spin+e*e)-e*e*(a_spin-lambda)**two 

              b0 = four*r_tp1**three*ar(1)+three*ar(2)*r_tp1**two+two*ar(3)*r_tp1+ar(4)
              b1 = two*r_tp1**two*ar(1)+ar(2)*r_tp1+ar(3)/three
              b2 = four/three*r_tp1*ar(1)+ar(2)/three
              b3 = ar(1)
              g2 = three/four*(b1*b1-b0*b2)
              g3 = (three*b0*b1*b2-two*b1*b1*b1-b0*b0*b3)/16.D0
 
              tp1 = infinity
              If(r_ini-r_tp1.ne.zero)then 
                  tobs=b0/four/(r_ini-r_tp1)+b1/four
              else
                  tobs=infinity
              endif 
              If(rhorizon-r_tp1.ne.zero)then
                  thorizon=b1/four+b0/four/(rhorizon-r_tp1)
              else
                  thorizon=infinity  
              endif 
              tp2=b0/four/(r_tp2-r_tp1)+b1/four 
              tinf=b1/four      
              h=-b1/four 

              call root3(zero,-g2/four,-g3/four,dd(1),dd(2),dd(3),del)    

              index_p5(1)=0
              cases_int=1
              integ15(1)=1.D100
              call weierstrass_int_J3(tobs,infinity,dd,del,a4,b4,index_p5,rff_p,integ05,cases_int) 
              PI0=integ05(1) 
              select case(cases)
              CASE(1)
                  If(kvecr .ge. zero)then !**particle will goto infinity.
                      index_p5(1)=0
                      cases_int=1
                      call weierstrass_int_J3(tinf,tobs,dd,del,a4,b4,index_p5,rff_p,integ05,cases_int)
                      p_total = integ05(1)   
                  ELSE 
                      If(.not.indrhorizon)then
                          index_p5(1)=0
                          cases_int=1 
                          call weierstrass_int_J3(tinf,infinity,dd,del,a4,b4,index_p5,rff_p,integ15,cases_int)
                          p_total = PI0+integ15(1) 
                      ELSE      !kvecr<0, photon will fall into black hole unless something encountered. 
                          index_p5(1)=0  
                          cases_int=1
                          call weierstrass_int_J3(tobs,thorizon,dd,del,a4,b4,index_p5,rff_p,integ05,cases_int)
                          p_total = integ05(1) 
                      ENDIF
                  ENDIF  
              CASE(2)
                  If(.not.indrhorizon)then 
                      index_p5(1)=0
                      cases_int=1  
                      call weierstrass_int_J3(tp2,infinity,dd,del,a4,b4,index_p5,rff_p,integ15,cases_int) 
                      p_tp1_tp2=integ15(1) 
                      p_total = 5.D0*p_tp1_tp2 
                  else   !photon has probability to fall into black hole.
                      If(kvecr.le.zero)then
                          index_p5(1)=0
                          cases_int=1
                          call weierstrass_int_J3(tobs,thorizon,dd,del,a4,b4,index_p5,rff_p,integ05,cases_int)
                          p_total = integ05(1) 
                      ELSE  !p_r>0, photon will meet the r_tp2 turning point and turn around then goto vevnt horizon.     
                          index_p5(1)=0
                          cases_int=1 
                          call weierstrass_int_J3(tp2,tobs,dd,del,a4,b4,index_p5,rff_p,integ05,cases_int)  
                          call weierstrass_int_J3(tp2,thorizon,dd,del,a4,b4,index_p5,rff_p,integ15,cases_int) 
                          p_total = integ15(1)+integ05(1) 
                      ENDIF
                  ENDIF                             
              END SELECT     
!************************************************************************************************        
          ELSE   !equation R(r)=0 has no real roots. we use the Jacobi's elliptic 
                 !integrations and functions to compute the integrations.
              IF(mve.lt.one)THEN
                  u=real(bb(4))
                  w=abs(aimag(bb(4)))
                  v=real(bb(3))
                  s=abs(aimag(bb(3)))
                  ac1=s*s
                  bc1=w*w+s*s+(u-v)*(u-v)
                  cc1=w*w 
                  L1=(bc1+dsqrt(bc1*bc1-four*ac1*cc1))/two/ac1            
                  L2=(bc1-dsqrt(bc1*bc1-four*ac1*cc1))/two/ac1 
                  alpha1=(L1*v-u)/(L1-one)
                  alpha2=(L2*v-u)/(L2-one)
                  thorizon = dsqrt((L1-one)/(L1-L2))*(rhorizon-alpha1)/dsqrt((rhorizon-v)**2+ac1)  
                  tobs = dsqrt((L1-one)/(L1-L2))*(r_ini-alpha1)/dsqrt((r_ini-v)**2+ac1)  
                  t_inf = dsqrt((L1-one)/(L1-L2))
                  m2 = (L1-L2)/L1   

                  pinf = EllipticF(tobs,m2)
                  IF(kvecr.lt.zero)THEN
                      p_total = ( pinf-EllipticF(thorizon,m2) )/s/dsqrt(-come*L1)   
                  ELSE
                      p_total = ( EllipticF(t_inf,m2)-pinf )/s/dsqrt(-come*L1)  
                  ENDIF                  
              ENDIF   
              IF(mve.gt.one)THEN
                  u=real(bb(4))
                  w=abs(aimag(bb(4)))
                  v=real(bb(3))
                  s=abs(aimag(bb(3)))
                  ac1=s*s
                  bc1=-w*w-s*s-(u-v)*(u-v)
                  cc1=w*w 
                  L1=(bc1+dsqrt(bc1*bc1-four*ac1*cc1))/two/ac1            
                  L2=(bc1-dsqrt(bc1*bc1-four*ac1*cc1))/two/ac1 
                  alpha1=(L1*v+u)/(L1+one)
                  alpha2=(L2*v+u)/(L2+one)  
                  thorizon = dsqrt( (L1-L2)/(-L2*(one+L1)) )*(rhorizon-alpha1)/dsqrt( (rhorizon-u)**2+cc1 ) 
                  tobs = dsqrt( (L1-L2)/(-L2*(one+L1)) )*(r_ini-alpha1)/dsqrt( (r_ini-u)**2+cc1 )  
                  t_inf = dsqrt( (L1-L2)/(-L2*(one+L1)) ) 
                  m2 = (L2-L1)/L2
 
                  pinf = EllipticF(tobs,m2)    
                  c_temp = L2*(one+L1)/(L1-L2)
                  If(kvecr.lt.zero)then
                      p_total = ( abs(pinf)-EllipticF(thorizon,m2) )/s/dsqrt(-come*L2)       
                  else
                      p_total = ( EllipticF(t_inf,m2)-abs(pinf) )/s/dsqrt(-come*L2) 
                  endif                    
              ENDIF       
          ENDIF     
      RETURN
      END FUNCTION p_total


!**************************************************************************************************************
      FUNCTION p_total_MB(kvecr,kvect,lambda,q,mve,ep,sin_ini,cos_ini,a_spin,e,r_ini)
!**************************************************************************************************************
!*     PURPOSE:  Computes the integral value of \int^r dr (R)^{-1/2}, from the starting position to
!*               the termination----either at infinity or at the event horizon. The constant of motion
!*               mve = 1.  
!*     INPUTS:   kvecr-------------k_{r}, initial r component of four momentum of a particle
!*                                 measured under an LNRF. See equation (93).
!*               kvect-------------k_{\theta}, initial \theta component of four momentum of a particle
!*                                 measured under an LNRF. See equation (94). 
!*               lambda,q,mve,ep-------constants of motion, defined by lambda=L_z/E, q=Q/E^2,
!*                                     mve = \mu_m/E, ep=epsilon/varepsilon. 
!*               sin_ini,cos_ini-------sin_ini=sin(\theta_{ini}), cos_ini=cos(\theta_{ini}), where 
!*                                     \theta_{ini} is the initial \theta coordinate of the particle.
!*               a_spin,e--------------the spin and the electric charge of the black hole, 
!*                                     and -1 =< a_spin^2+e^2 <= 1 must be satisfied. 
!*               r_ini-----------------the initial radial coordinate of the particle.    
!*     OUTPUTS:  p_total--------which is the value of integrals \int^r dr (R)^{-1/2}, along a 
!*                              whole geodesic, that is from the starting position to either go to
!*                              infinity or fall in to black hole.          
!*     ROUTINES CALLED: root3, weierstrass_int_J3, radiustp, weierstrassP, EllipticF, carlson_doublecomplex5 
!*     ACCURACY:   Machine.    
!*     AUTHOR:     Yang & Wang (2012)  
!*     DATE WRITTEN:  5 Jan 2012
!*     REVISIONS: ******************************************  
      USE constants
      IMPLICIT NONE
      DOUBLE PRECISION :: p_total_MB,kvecr,kvect,lambda,q,mve,sin_ini,cos_ini,a_spin,r_ini,&
             phyr,timer,affr,r_coord,integ05(5),integ5(5),integ15(5),&
             b0,b1,b2,b3,g2,g3,tp1,tp2,tobs,thorizon,tinf,rhorizon,rff_p,a4,b4,&
             PI0_obs_inf,r_tp1,r_tp2,PI0,PI0_total,PI1_p,PI2_p,PI1_phi,PI2_phi,&
             PI1_time,PI2_time,PI1_aff,PI2_aff,rp,rm,k1,k2,& 
             wp,wm,hp,hm,wbarp,wbarm,tp,pp,p1,p2,PI01,PI2,p_temp,h,pp_aff,&
             pp_time,pp_phi,time_temp,E_add,E_m,p1_phi,p1_time,p1_aff,p2_phi,&
             p2_time,p2_aff,ac1,bc1,cc1,e,ep,Atp,Atm,App,Apm,&
             ep_1,e_1,trp,trm    
      COMPLEX*16 :: dd(3),bb(4)
      INTEGER :: del,index_p5(5),cases_int,reals,cases,count_num=1,t1,t2,PI0_obs_hori,&
              i,j
      LOGICAL :: r_ini_eq_rtp,indrhorizon,r1eqr2  
 
       !**************************************  
          rp=one+sqrt(one-a_spin**two-e*e) 
          rhorizon=rp  

          b4=one
          a4=zero 
          r_ini_eq_rtp=.false.
          indrhorizon=.false.
          call radiustp(kvecr,a_spin,e,r_ini,lambda,q,mve,ep,r_tp1,r_tp2,&
                                     reals,r_ini_eq_rtp,indrhorizon,r1eqr2,cases,bb)   
       !********************************************************************        
          b0=two*(one+e*ep) ! we assume |e*ep|<1, thus b0>0
          b1=-(lambda*lambda+q+e*e*(one-ep*ep))/three
          b2=two*(q+(lambda-a_spin)*(lambda-a_spin)+a_spin*e*ep*(a_spin-lambda))/three
          b3=-q*(a_spin*a_spin+e*e)-e*e*(a_spin-lambda)**two
          g2 = three/four*(b1*b1-b0*b2)
          g3 = (three*b0*b1*b2-two*b1*b1*b1-b0*b0*b3)/16.D0    
        
          call root3(zero,-g2/four,-g3/four,dd(1),dd(2),dd(3),del)

          tp1 = b0/four*r_tp1+b1/four
          IF(r_tp2.LT.infinity)THEN
              tp2 = b0/four*r_tp2+b1/four
          ELSE
              tp2 = infinity
          ENDIF 
          thorizon = b0/four*rhorizon+b1/four
          tobs = b0/four*r_ini+b1/four 
         
          index_p5(1)=0
          cases_int=1
          call weierstrass_int_J3(tobs,infinity,dd,del,a4,b4,index_p5,rff_p,integ05,cases_int) 
          call weierstrass_int_J3(tp1,tobs,dd,del,a4,b4,index_p5,rff_p,integ15,cases_int) 
          call weierstrass_int_J3(tobs,tp2,dd,del,a4,b4,index_p5,rff_p,integ5,cases_int) 
          PI0=integ05(1) 
          PI1_p=integ15(1)  
          PI2_p=integ5(1) 
          SELECT CASE(cases)  
          case(1)
              IF(kvecr .ge. zero)THEN
                  p_total_MB = PI0             
              ELSE
                  If(.not.indrhorizon)then 
                      p_total_MB = two*PI1_p+PI2_p   
                  ELSE    !kvecr<0, photon will fall into black hole unless something encountered. 
                      index_p5(1)=0  
                      cases_int=1
                      call weierstrass_int_J3(thorizon,tobs,dd,del,a4,b4,index_p5,rff_p,integ05,cases_int)
                      p_total_MB = integ05(1)  
                  ENDIF 
              ENDIF
          case(2)
              If(.not.indrhorizon)then
                  p_total_MB = four*(PI1_p+PI2_p)  
              else   !photon has probability to fall into black hole.
                  If(kvecr.le.zero)then
                      index_p5(1)=0
                      cases_int=1
                      call weierstrass_int_J3(thorizon,tobs,dd,del,a4,b4,index_p5,rff_p,integ05,cases_int)
                      p_total_MB = integ05(1)  
                  ELSE  !p_r>0, photon will meet the r_tp2 turning point and turn around then goto vevnt horizon.     
                      index_p5(1)=0
                      cases_int=1   
                      call weierstrass_int_J3(thorizon,tp2,dd,del,a4,b4,index_p5,rff_p,integ15,cases_int) 
                      p_total_MB = PI2_p+integ15(1)  
                  ENDIF
              ENDIF 
          END SELECT  
      RETURN 
      END FUNCTION p_total_MB 
!*******************************************************************************
      end module blcoordinates
!*******************************************************************************



!*******************************************************************************
      Module sigma2p_time2p
!*******************************************************************************
!*    this module aims on solve the equations t(p) = t_0, and \sigma(p) = \sigma_0
!*    to get the roots p_0 by bisection or iterative method.      
!*******************************************************************************
      use blcoordinates
      implicit none
 
      contains
!*******************************************************************************
      subroutine coeff_of_p(kp,r_ini,sin_ini,cos_ini,a_spin,e,lambda,&
                               q,mve,ep,coeff_sigma,coeff_time)
!**************************************************************************************************************
!*     PURPOSE:  to compute the coefficients \sigma_t and \sigma_s defined in table 7. 
!*     INPUTS:   kp(4)-------------An array contains k_{r}, k_{\theta}, k_{\phi}, k_{t}, they are the 
!*                                 initial components of four momentum of a particle
!*                                 measured under an LNRF. See equations (91)=(95). 
!*               r_ini-----------------the initial radial coordinate of the particle. 
!*               lambda,q,mve,ep-------constants of motion, defined by lambda=L_z/E, q=Q/E^2,
!*                                     mve = \mu_m/E, ep=epsilon/varepsilon. 
!*               sin_ini,cos_ini-------sin_ini=sin(\theta_{ini}), cos_ini=cos(\theta_{ini}), where 
!*                                     \theta_{ini} is the initial \theta coordinate of the particle.
!*               a_spin,e--------------the spin and the electric charge of the black hole, 
!*                                     and -1 =< a_spin^2+e^2 <= 1 must be satisfied.    
!*     OUTPUTS:  coeff_sigma, coeff_time.          
!*     ROUTINES CALLED: NONE
!*     ACCURACY:   Machine.    
!*     AUTHOR:     Yang & Wang (2013)  
!*     DATE WRITTEN:  5 Jan 2013
!*     REVISIONS: ******************************************  
      use constants  
      use blcoordinates
      implicit none 
      double precision kp(1:4),r_ini,sin_ini,cos_ini,a_spin,e,&
             lambda,q,mve,ep,a2,e2,rp,rm,k1,k2,b1,b0
      double precision coeff_p,coeff_mu,coeff_r,coeff_sigma,coeff_time
      double precision r_tp1,r_tp2,mu_tp1,mu_tp2,Atp,Atm
      complex*16 bb(1:4)
      integer :: reals_r,reals_mu,cases
      logical :: r_ini_eq_rtp,indrhorizon,r1eqr2,mobseqmtp

      call mutp(kp(3),kp(2),sin_ini,cos_ini,a_spin,lambda,q,mve,mu_tp1,mu_tp2,reals_mu,mobseqmtp)
      call radiustp(kp(1),a_spin,e,r_ini,lambda,q,mve,ep,r_tp1,&
                           r_tp2,reals_r,r_ini_eq_rtp,indrhorizon,r1eqr2,cases,bb)

      a2 = a_spin*a_spin
      e2 = e*e
      rp = one+dsqrt(one-a2-e2)
      rm = one-dsqrt(one-a2-e2)
      k1 = eight - two*a_spin*lambda+four*(e*ep-e2)-e*e*e*ep
      k2 = e2*(e2+a_spin*lambda)-two*(e2+a2)*(two+e*ep)
      If(mve.ne.1)then
          If(reals_r.ne.0)then
              Atp = ( k1*rp + k2 )/(rp-rm)/(r_tp1-rp)
              Atm = ( k1*rm + k2 )/(rp-rm)/(r_tp1-rm)
              coeff_sigma = a2*mu_tp1*mu_tp1 + r_tp1*r_tp1
              coeff_time = coeff_sigma + ( (two+e*ep)*(two+r_tp1) - e2 ) 
              If(rp.ne.rm)then
                  coeff_time = coeff_time + Atp-Atm
              else if(rp.eq.rm)then
                  coeff_time = coeff_time + k1/(r_tp1-rp)+(k1*rp+k2)/(r_tp1-rp)**two
              endif
          else
              coeff_sigma = a2*mu_tp1*mu_tp1
              coeff_time = coeff_sigma + ( (two+e*ep)*two- e2 )/dsqrt(one-mve*mve)
          endif
      else
          b0 = two*(one+e*ep)
          b1 = -(q+lambda*lambda+e2*(one-ep*ep))/three
          coeff_sigma = b1*b1+a2*mu_tp1*mu_tp1/two
          coeff_time = coeff_sigma + (two+e*ep)*(two-b1/b0)-e2
      endif
 
      Return
      end subroutine coeff_of_p


!*******************************************************************************
      Function Func_temp_time(p,time_0,kp,r_ini,sin_ini,cos_ini,&
                            a_spin,e,lambda,q,mve,ep,coeff_time)
!*********************************************************************************************** 
!*     PURPOSE:  to compute the Function f_{t}(p)=(time_0-\bar{t}(p))/coeff_time, 
!*               defined by equation (B.4) of our paper.
!*     INPUTS:   p-----------------The independent variable, which must be positive.
!*               time_0------------The equation t(p) = time_0.
!*               kp(4)-------------An array contains k_{r}, k_{\theta}, k_{\phi}, k_{t}, they are the 
!*                                 initial components of four momentum of a particle
!*                                 measured under an LNRF. See equations (91)=(95). 
!*               r_ini-----------------the initial radial coordinate of the particle. 
!*               lambda,q,mve,ep-------constants of motion, defined by lambda=L_z/E, q=Q/E^2,
!*                                     mve = \mu_m/E, ep=epsilon/varepsilon. 
!*               sin_ini,cos_ini-------sin_ini=sin(\theta_{ini}), cos_ini=cos(\theta_{ini}), where 
!*                                     \theta_{ini} is the initial \theta coordinate of the particle.
!*               a_spin,e--------------the spin and the electric charge of the black hole, 
!*                                     and -1 =< a_spin^2+e^2 <= 1 must be satisfied.       
!*     ROUTINES CALLED: NONE
!*     ACCURACY:   Machine.    
!*     AUTHOR:     Yang & Wang (2013)  
!*     DATE WRITTEN:  5 Jan 2013
!*     REVISIONS: ****************************************** 
      use constants
      Use blcoordinates
      Implicit none 
      Double precision time_0,coeff_time,kp(1:4),a_spin,e,lambda,q,mve,ep,&
             r_ini,sin_ini,cos_ini,radi,mu,tim,phy,sigm,Func_temp_time,p,theta_star
      Logical :: cir_orbt

      cir_orbt = .FALSE.
      theta_star = zero
 
      CALL YNOGKM(p,kp,lambda,q,mve,ep,sin_ini,cos_ini,a_spin,e,r_ini,&
                                        radi,mu,tim,phy,sigm,cir_orbt,theta_star)
      ! equation (B.4) of Yang & Wang 2013, A&A.
      Func_temp_time = (time_0-(tim - coeff_time*p))/coeff_time 
 
      Return
      End Function Func_temp_time


!*******************************************************************************
      Function Func_temp_time_bs(p,time_0,kp,r_ini,sin_ini,cos_ini,&
                            a_spin,e,lambda,q,mve,ep,coeff_time)
!*********************************************************************************************** 
!*     PURPOSE:  to compute the Function f_{t}(p)=time_0-t(p), which is an temporary function, 
!*               which is used in the bisection method to solve the equation t(p) = time_0.
!*     INPUTS:   p-----------------The independent variable, which must be positive.
!*               time_0------------The equation t(p) = time_0.
!*               kp(4)-------------An array contains k_{r}, k_{\theta}, k_{\phi}, k_{t}, they are the 
!*                                 initial components of four momentum of a particle
!*                                 measured under an LNRF. See equations (91)=(95). 
!*               r_ini-----------------the initial radial coordinate of the particle. 
!*               lambda,q,mve,ep-------constants of motion, defined by lambda=L_z/E, q=Q/E^2,
!*                                     mve = \mu_m/E, ep=epsilon/varepsilon. 
!*               sin_ini,cos_ini-------sin_ini=sin(\theta_{ini}), cos_ini=cos(\theta_{ini}), where 
!*                                     \theta_{ini} is the initial \theta coordinate of the particle.
!*               a_spin,e--------------the spin and the electric charge of the black hole, 
!*                                     and -1 =< a_spin^2+e^2 <= 1 must be satisfied.   
!*     ROUTINES CALLED: NONE
!*     ACCURACY:   Machine.    
!*     AUTHOR:     Yang & Wang (2013)  
!*     DATE WRITTEN:  5 Jan 2013
!*     REVISIONS: ****************************************** 
      use constants
      Use blcoordinates
      Implicit none 
      Double precision time_0,coeff_time,kp(1:4),a_spin,e,lambda,q,mve,ep,&
             r_ini,sin_ini,cos_ini,radi,mu,tim,phy,sigm,Func_temp_time_bs,p,theta_star
      Logical :: cir_orbt

      cir_orbt = .FALSE.
      theta_star = zero
 
      CALL YNOGKM(p,kp,lambda,q,mve,ep,sin_ini,cos_ini,a_spin,e,r_ini,&
                                        radi,mu,tim,phy,sigm,cir_orbt,theta_star)
      Func_temp_time_bs = time_0-tim
 
      Return
      End Function Func_temp_time_bs


!*******************************************************************************
      Function Func_temp_sigma(p,sigma_0,kp,r_ini,sin_ini,cos_ini,&
                             a_spin,e,lambda,q,mve,ep,coeff_sigma)
!*********************************************************************************************** 
!*     PURPOSE:  to compute the Function f_{\sigma}(p)=(sigma_0-\bar{\sigma}(p))/coeff_time, 
!*               defined by equation (B.3) of our paper.
!*     INPUTS:   p-----------------The independent variable, which must be positive.
!*               sigma_0------------The equation \sigma(p) = sigma_0.
!*               kp(4)-------------An array contains k_{r}, k_{\theta}, k_{\phi}, k_{t}, they are the 
!*                                 initial components of four momentum of a particle
!*                                 measured under an LNRF. See equations (91)=(95). 
!*               r_ini-----------------the initial radial coordinate of the particle. 
!*               lambda,q,mve,ep-------constants of motion, defined by lambda=L_z/E, q=Q/E^2,
!*                                     mve = \mu_m/E, ep=epsilon/varepsilon. 
!*               sin_ini,cos_ini-------sin_ini=sin(\theta_{ini}), cos_ini=cos(\theta_{ini}), where 
!*                                     \theta_{ini} is the initial \theta coordinate of the particle.
!*               a_spin,e--------------the spin and the electric charge of the black hole, 
!*                                     and -1 =< a_spin^2+e^2 <= 1 must be satisfied.    
!*     OUTPUTS:  Func_temp_sigma      
!*     ROUTINES CALLED: NONE
!*     ACCURACY:   Machine.    
!*     AUTHOR:     Yang & Wang (2013)  
!*     DATE WRITTEN:  5 Jan 2013
!*     REVISIONS: ******************************************
      use constants
      Use blcoordinates
      Implicit none 
      Double precision sigma_0,coeff_sigma,kp(1:4),a_spin,e,lambda,q,mve,ep,&
             r_ini,sin_ini,cos_ini,radi,mu,tim,phy,sigm,Func_temp_sigma,p,theta_star
      Logical :: cir_orbt

      cir_orbt = .FALSE.
      theta_star = zero

      CALL YNOGKM(p,kp,lambda,q,mve,ep,sin_ini,cos_ini,a_spin,e,r_ini,&
                                        radi,mu,tim,phy,sigm,cir_orbt,theta_star)
      ! equation (B.3) of Yang & Wang 2013, A&A.
      Func_temp_sigma = (sigma_0-(sigm - coeff_sigma*p))/coeff_sigma
 
      Return
      End Function Func_temp_sigma


!*******************************************************************************
      Function Func_temp_sigma_bs(p,sigma_0,kp,r_ini,sin_ini,cos_ini,&
                             a_spin,e,lambda,q,mve,ep,coeff_sigma)
!*********************************************************************************************** 
!*     PURPOSE:  to compute the Function f_{t}(p)=sigma_0-sigma(p), which is an temporary function, 
!*               which is used in the bisection method to solve the equation \sigma(p) = sigma_0.
!*     INPUTS:   p-----------------The independent variable, which must be positive.
!*               sigma_0-----------The equation \sigma(p) = \sigma_0.
!*               kp(4)-------------An array contains k_{r}, k_{\theta}, k_{\phi}, k_{t}, they are the 
!*                                 initial components of four momentum of a particle
!*                                 measured under an LNRF. See equations (91)=(95). 
!*               r_ini-----------------the initial radial coordinate of the particle. 
!*               lambda,q,mve,ep-------constants of motion, defined by lambda=L_z/E, q=Q/E^2,
!*                                     mve = \mu_m/E, ep=epsilon/varepsilon. 
!*               sin_ini,cos_ini-------sin_ini=sin(\theta_{ini}), cos_ini=cos(\theta_{ini}), where 
!*                                     \theta_{ini} is the initial \theta coordinate of the particle.
!*               a_spin,e--------------the spin and the electric charge of the black hole, 
!*                                     and -1 =< a_spin^2+e^2 <= 1 must be satisfied.   
!*     ROUTINES CALLED: NONE
!*     ACCURACY:   Machine.    
!*     AUTHOR:     Yang & Wang (2013)  
!*     DATE WRITTEN:  5 Jan 2013
!*     REVISIONS: ****************************************** 
      use constants
      Use blcoordinates
      Implicit none 
      Double precision sigma_0,coeff_sigma,kp(1:4),a_spin,e,lambda,q,mve,ep,&
             r_ini,sin_ini,cos_ini,radi,mu,tim,phy,sigm,Func_temp_sigma_bs,p,theta_star
      Logical :: cir_orbt

      cir_orbt = .FALSE.
      theta_star = zero

      CALL YNOGKM(p,kp,lambda,q,mve,ep,sin_ini,cos_ini,a_spin,e,r_ini,&
                                        radi,mu,tim,phy,sigm,cir_orbt,theta_star)
      ! equation (B.3) of Yang & Wang 2013, A&A.
      Func_temp_sigma_bs =  sigma_0-sigm  
 
      Return
      End Function Func_temp_sigma_bs


!*******************************************************************************
      Function time2p_bisection(time_0,kp,r_ini,sin_ini,cos_ini,&
                     a_spin,e,lambda,q,mve,ep)
!*********************************************************************************************** 
!*     PURPOSE:  to solve the equation t(p) = time_0 by using the bisection method. The value p_0 is 
!*               returned which satisfy t(p_0) = time_0.
!*     INPUTS:   time_0------------The equation t(p) = time_0.
!*               kp(4)-------------An array contains k_{r}, k_{\theta}, k_{\phi}, k_{t}, they are the 
!*                                 initial components of four momentum of a particle
!*                                 measured under an LNRF. See equations (91)=(95). 
!*               r_ini-----------------the initial radial coordinate of the particle. 
!*               lambda,q,mve,ep-------constants of motion, defined by lambda=L_z/E, q=Q/E^2,
!*                                     mve = \mu_m/E, ep=epsilon/varepsilon. 
!*               sin_ini,cos_ini-------sin_ini=sin(\theta_{ini}), cos_ini=cos(\theta_{ini}), where 
!*                                     \theta_{ini} is the initial \theta coordinate of the particle.
!*               a_spin,e--------------the spin and the electric charge of the black hole, 
!*                                     and -1 =< a_spin^2+e^2 <= 1 must be satisfied.    
!*     OUTPUTS:  time2p_bisection.      
!*     ROUTINES CALLED: NONE
!*     ACCURACY:   Machine.    
!*     AUTHOR:     Yang & Wang (2013)  
!*     DATE WRITTEN:  5 Jan 2013
!*     REVISIONS: ****************************************** 
      use blcoordinates
      implicit none
      !double precision, external :: Func_temp_sigma,Func_temp_time
      double precision p,time_0,r_ini,sin_ini,cos_ini,&
             a_spin,e,lambda,q,mve,ep,coeff_sigma,p_temp,&
             coeff_time,time2p_bisection,kp(1:4),p1,p2,p_ini,Det_p,&
             del_p1,del_p2,F_p1,F_p2,F_pc,pc,F_temp
      integer counts

      If(time_0.le.1.D-10)then
          time2p_bisection = zero
          return
      Else
          counts = 1 
          p_ini = 1.D-5
 
          p1 = p_ini
          F_p1 = Func_temp_time_bs(p1,time_0,kp,r_ini,sin_ini,cos_ini,&
                                    a_spin,e,lambda,q,mve,ep,coeff_time)
          p2 = p1
          F_p2 = F_p1
     
          Do while(F_p2.gt.zero) 
              F_temp = F_p2
              p2 = p2 + 1.D-4 
              F_p2 = Func_temp_time_bs(p2,time_0,kp,r_ini,sin_ini,cos_ini,&
                                    a_spin,e,lambda,q,mve,ep,coeff_time) 
          Enddo 
          p1 = p2-1.D-4
          F_p1 = F_temp
    
          Do while(dabs(p1-p2).lt.1.D-6)
              pc = (p1+p2)*half
              F_pc = Func_temp_time_bs(pc,time_0,kp,r_ini,sin_ini,cos_ini,&
                                    a_spin,e,lambda,q,mve,ep,coeff_time)
              If(F_pc*F_p1 .gt. zero)then
                  p1 = pc
                  F_p1 = F_pc
              else
                  p2 = pc 
                  F_p2 = F_pc
              endif 
          Enddo 
          time2p_bisection = (p1+p2)*half
      Endif
      end function time2p_bisection


!*******************************************************************************
      Function sigma2p_bisection(sigma_0,kp,r_ini,sin_ini,cos_ini,&
                     a_spin,e,lambda,q,mve,ep)
!*********************************************************************************************** 
!*     PURPOSE:  to solve the equation sigma(p) = sigma_0 by using the bisection method. The value p_0 is 
!*               returned which satisfy sigma(p_0) = sigma_0.
!*     INPUTS:   sigma_0------------The equation sigma(p) = sigma_0.
!*               kp(4)-------------An array contains k_{r}, k_{\theta}, k_{\phi}, k_{t}, they are the 
!*                                 initial components of four momentum of a particle
!*                                 measured under an LNRF. See equations (91)=(95). 
!*               r_ini-----------------the initial radial coordinate of the particle. 
!*               lambda,q,mve,ep-------constants of motion, defined by lambda=L_z/E, q=Q/E^2,
!*                                     mve = \mu_m/E, ep=epsilon/varepsilon. 
!*               sin_ini,cos_ini-------sin_ini=sin(\theta_{ini}), cos_ini=cos(\theta_{ini}), where 
!*                                     \theta_{ini} is the initial \theta coordinate of the particle.
!*               a_spin,e--------------the spin and the electric charge of the black hole, 
!*                                     and -1 =< a_spin^2+e^2 <= 1 must be satisfied.          
!*     ROUTINES CALLED: NONE
!*     ACCURACY:   Machine.    
!*     AUTHOR:     Yang & Wang (2013)  
!*     DATE WRITTEN:  5 Jan 2013
!*     REVISIONS: ****************************************** 
      use blcoordinates
      implicit none 
      double precision p,sigma_0,r_ini,sin_ini,cos_ini,&
             a_spin,e,lambda,q,mve,ep,coeff_sigma,p_temp,&
             coeff_time,sigma2p_bisection,kp(1:4),p1,p2,p_ini,Det_p,&
             del_p1,del_p2,F_p1,F_p2,F_pc,pc,F_temp
      integer counts

      If(sigma_0.le.1.D-10)then
          sigma2p_bisection = zero
          return
      Else
          counts = 1 
          p_ini = 1.D-5
 
          p1 = p_ini
          F_p1 = Func_temp_sigma_bs(p1,sigma_0,kp,r_ini,sin_ini,cos_ini,&
                             a_spin,e,lambda,q,mve,ep,coeff_sigma) 

          p2 = p1
          F_p2 = F_p1
     
          Do while(F_p2.gt.zero) 
              F_temp = F_p2
              p2 = p2 + 1.D-4 
              F_p2 = Func_temp_sigma_bs(p2,sigma_0,kp,r_ini,sin_ini,cos_ini,&
                                    a_spin,e,lambda,q,mve,ep,coeff_time) 
          Enddo 
          p1 = p2-1.D-4
          F_p1 = F_temp
    
          Do while(dabs(p1-p2).lt.1.D-6)
              pc = (p1+p2)*half
              F_pc = Func_temp_sigma_bs(pc,sigma_0,kp,r_ini,sin_ini,cos_ini,&
                                    a_spin,e,lambda,q,mve,ep,coeff_time)
              If(F_pc*F_p1 .gt. zero)then
                  p1 = pc
                  F_p1 = F_pc
              else
                  p2 = pc 
                  F_p2 = F_pc
              endif 
          Enddo 
          sigma2p_bisection = (p1+p2)*half
      Endif
      end function sigma2p_bisection


!**************************************************************
      Function time2p(time_0,kp,r_ini,sin_ini,cos_ini,&
                     a_spin,e,lambda,q,mve,ep)
!*********************************************************************************************** 
!*     PURPOSE:  to solve the equation t(p) = time_0 by using the iterative method. The value p_0 is 
!*               returned which satisfy t(p_0) = time_0.
!*     INPUTS:   time_0------------The equation t(p) = time_0.
!*               kp(4)-------------An array contains k_{r}, k_{\theta}, k_{\phi}, k_{t}, they are the 
!*                                 initial components of four momentum of a particle
!*                                 measured under an LNRF. See equations (91)=(95). 
!*               r_ini-----------------the initial radial coordinate of the particle. 
!*               lambda,q,mve,ep-------constants of motion, defined by lambda=L_z/E, q=Q/E^2,
!*                                     mve = \mu_m/E, ep=epsilon/varepsilon. 
!*               sin_ini,cos_ini-------sin_ini=sin(\theta_{ini}), cos_ini=cos(\theta_{ini}), where 
!*                                     \theta_{ini} is the initial \theta coordinate of the particle.
!*               a_spin,e--------------the spin and the electric charge of the black hole, 
!*                                     and -1 =< a_spin^2+e^2 <= 1 must be satisfied.     
!*     ROUTINES CALLED: NONE
!*     ACCURACY:   Machine.    
!*     AUTHOR:     Yang & Wang (2013)  
!*     DATE WRITTEN:  5 Jan 2013
!*     REVISIONS: ****************************************** 
      use blcoordinates
      implicit none
      double precision p,time_0,r_ini,sin_ini,cos_ini,&
             a_spin,e,lambda,q,mve,ep,coeff_sigma,p_temp,&
             coeff_time,time2p,kp(1:4),p1,p2,p_ini,Det_p,&
             del_p1,del_p2,F_p1,F_p2
      integer counts

      If(time_0.le.1.D-10)then
          time2p = zero
          return
      Else
          counts = 1
          call coeff_of_p(kp,r_ini,sin_ini,cos_ini,a_spin,e,lambda,&
                             q,mve,ep,coeff_sigma,coeff_time) 
          p = 1.D-6 
          Do while(.True.) 
              counts = counts+1
              p_temp = Func_temp_time(p,time_0,kp,r_ini,sin_ini,cos_ini,&
                                  a_spin,e,lambda,q,mve,ep,coeff_time)
              If(dabs(p-p_temp).lt.1.D-11)exit 
              p = p_temp  
          Enddo 
          time2p = p_temp
          Return       
      Endif
      end function time2p


!**************************************************************
      Function sigma2p(sigma,kp,r_ini,sin_ini,cos_ini,&
                     a_spin,e,lambda,q,mve,ep) 
!*********************************************************************************************** 
!*     PURPOSE:  to solve the equation \sigma(p) = sigma_0 by using the iterative method. The value p_0 is 
!*               returned which satisfy \sigma(p_0) = sigma_0.
!*     INPUTS:   sigma_0-----------The equation \sigma(p_0) = sigma_0.
!*               kp(4)-------------An array contains k_{r}, k_{\theta}, k_{\phi}, k_{t}, they are the 
!*                                 initial components of four momentum of a particle
!*                                 measured under an LNRF. See equations (91)=(95). 
!*               r_ini-----------------the initial radial coordinate of the particle. 
!*               lambda,q,mve,ep-------constants of motion, defined by lambda=L_z/E, q=Q/E^2,
!*                                     mve = \mu_m/E, ep=epsilon/varepsilon. 
!*               sin_ini,cos_ini-------sin_ini=sin(\theta_{ini}), cos_ini=cos(\theta_{ini}), where 
!*                                     \theta_{ini} is the initial \theta coordinate of the particle.
!*               a_spin,e--------------the spin and the electric charge of the black hole, 
!*                                     and -1 =< a_spin^2+e^2 <= 1 must be satisfied.          
!*     ROUTINES CALLED: NONE
!*     ACCURACY:   Machine.    
!*     AUTHOR:     Yang & Wang (2013)  
!*     DATE WRITTEN:  5 Jan 2013
!*     REVISIONS: ****************************************** 
      use blcoordinates
      implicit none
      !double precision, external :: Func_temp_sigma,Func_temp_time
      double precision p,sigma,r_ini,sin_ini,cos_ini,&
             a_spin,e,lambda,q,mve,ep,coeff_sigma,p_temp,&
             coeff_time,sigma2p,kp(1:4)

      If(sigma.le.1.D-10)then
          sigma2p = zero
          return
      Else
          p = 0.0000001D0 
          call coeff_of_p(kp,r_ini,sin_ini,cos_ini,a_spin,e,lambda,&
                             q,mve,ep,coeff_sigma,coeff_time)      
      
          Do while(.True.)
              p_temp = Func_temp_sigma(p,sigma,kp,r_ini,sin_ini,cos_ini,&
                                    a_spin,e,lambda,q,mve,ep,coeff_sigma)  
              If(dabs(p-p_temp).lt.1.D-14)exit 
              p = p_temp  
          Enddo

          sigma2p = p_temp
          Return
      Endif
      end function sigma2p


!***********************************************************
      END MODULE sigma2p_time2p
!*********************************************************** 


