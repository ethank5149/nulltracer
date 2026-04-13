      include  '../ynogkm.f90' 
      PROGRAM  MAIN
      use constants	
      IMPLICIT NONE  
      Double precision a_spin,beta,r0,sbeta,cbeta,e
	 
      a_spin = 0.998D0
      r0 = 15.D0
 
      e = 0.D0   
      IF(e*e+a_spin*a_spin.ge.one)Then
      write(*,*)'e^2+a^2>=1, there can be no horizon. wrong and stoped.'
          stop
      ENDIF
      call reashift2(a_spin,e,r0)
      END

!***************************************************************************************************** 
      subroutine reashift2(a_spin,e,r0)
!*****************************************************************************************************
      use constants
      use blcoordinates
      implicit none
      Double precision a_spin,e,r0,theta_star,rhorizon,ur,um,up,ut,ut1,ut2,theta0,sin0,&
             cos0,somiga,expnu,exppsi,expmu1,expmu2,gtt,gtp,grr,gmm,gpp,DD,vptl(1:3),&
             velocity(1:3),p,pmax,lambda,q,mve,ep,kvec(1:4),ra,mua,timea,phya,sigmaa,&
             p_temp   
      integer  k,j,n 
      logical :: cir_orbt

      cir_orbt = .false. 
      theta_star = zero
 
      n=20  
      rhorizon=one+sqrt(one-a_spin**two-e*e)
 
      open(unit=15,file='rayx1.txt',status="replace")
      open(unit=16,file='rayz1.txt',status="replace")  	 
      ur = -0.35D0
      um = zero
      up = -0.025D0   	  			
      DO j=0,n-1 
          theta0 = j*90.D0/((n-1)*1.D0)-0.001D0
          write(*,*)'cobs=',theta0
          If(theta0.ne.90.D0)then 
              cos0=cos(theta0*dtor)
              sin0=sin(theta0*dtor)
          else
              cos0=zero
              sin0=one
          endif

          call metricg(r0,sin0,cos0,a_spin,e,somiga,expnu,exppsi,expmu1,expmu2)
          gtt = -expnu*expnu+somiga*somiga*exppsi*exppsi
          gtp = -somiga*exppsi*exppsi
          grr = expmu1*expmu1
          gmm = expmu2*expmu2
          gpp = exppsi*exppsi
          DD = gtp*gtp-gtt*(one+grr*ur*ur+gmm*um*um+gpp*up*up)
          ut1 = (-gtp+dsqrt(DD))/gtt
          ut2 = (-gtp-dsqrt(DD))/gtt
          ut = ut1
     ! see equation (77) of Yang & wang 2013, A&A.
          vptl(1) = expmu1/expnu*ur/abs(ut)
          vptl(2) = expmu2/expnu*um/abs(ut)
          vptl(3) = exppsi/expnu*(up/abs(ut)-somiga) 
          IF(dsqrt(vptl(1)**2+vptl(2)**2+vptl(3)**2).ge.1.D0)THEN
             write(*,*)'supper light speed.',dsqrt(vptl(1)**2+vptl(2)**2+vptl(3)**2)
             stop
          ENDIF
	  velocity(1)=zero
	  velocity(2)=zero
	  velocity(3)=zero
 
          call lambdaqm(vptl,sin0,cos0,a_spin,e,r0,one,velocity,lambda,q,mve,ep,kvec) 
          pmax = p_total(kvec,lambda,q,mve,ep,sin0,cos0,a_spin,e,r0) 
          p_temp = 100.D0
          Do k=0,800
              p=k*pmax*0.98D0/800 
              !call YNOGKM(p,kvec,lambda,q,mve,ep,sin0,cos0,a_spin,e,r0,&
              !                          ra,mua,timea,phya,sigmaa,cir_orbt,theta_star) 
              ra = radius(p,kvec(1),lambda,q,mve,ep,a_spin,e,r0)
              mua = mucos(p,kvec(3),kvec(2),lambda,q,mve,sin0,cos0,a_spin)

              If(mua.ge.0.D0 .and. ra .ge. rhorizon*1.D0 .and. p .le.p_temp)then  
                  write(unit=15,fmt=*)sqrt(ra**two+a_spin**two)*sqrt(one-mua**two) 
                  write(unit=16,fmt=*)ra*mua
              Else
                  If(p_temp.eq.100.D0)then
                      p_temp = p
                  endif
                  write(unit=15,fmt=*)sqrt(ra/ra-100.D0) 
                  write(unit=16,fmt=*)sqrt(ra/ra-100.D0)         
              ENDIF 
          Enddo
      Enddo 
      close(unit=15)	
      close(unit=16)
      close(unit=17) 
      return
      end subroutine reashift2
 




