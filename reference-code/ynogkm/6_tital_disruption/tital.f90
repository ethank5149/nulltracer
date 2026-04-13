      include '../ynogkm.f90'
      !include './sigma_time_2p.f90'
!******************************************************************
      program main
      use blcoordinates
      use sigma2p_time2p
      implicit none  
      integer nm,n1,m1
      parameter(n1=31,m1=101,nm=n1*m1)
      double precision :: kp(1:4),r_ini,sin_ini,cos_ini,a_spin,p_temp,&
             e,lambda,q,mve,ep,p,radi,mu,time_0,phy,sigma_0,theta_ini,&
             coeff_time,coeff_sigma,theta_star=0.D0
      double precision somiga,expnu,exppsi,expmu1,expmu2,vel(1:3),velo,vptl(1:3),&
             sign_ch,time_1,sigma_1,sigma_bar_p,time_bar_p
      double precision phi_ini,r_ball,the,phi1,xp(1:n1*m1),yp(1:n1*m1),zp(1:n1*m1),&
             x_c,y_c,z_c,xp1,yp1,zp1,t,p_t,tim,sigm,a2,phy_ini 
      Logical :: cir_orbt
      integer i,j,k 

      read(unit=5,fmt=*)a_spin,theta_ini,r_ini,phi_ini 

      If(theta_ini.ne.90.D0)then 
          cos_ini = dcos(theta_ini*dtor)
          sin_ini = dsin(theta_ini*dtor)
      else
          cos_ini = zero
          sin_ini = one
      endif

      e = 0.D0
      a2 = a_spin*a_spin
      sign_ch = one
      cir_orbt = .FALSE.
      call metricg(r_ini,sin_ini,cos_ini,a_spin,e,somiga,expnu,exppsi,expmu1,expmu2)
      vel(1)=0.0D0!expmu1/expnu*robs/(robs**(2.05D0/two)+a_spin)!*zero
      vel(2)=0.0D0!expmu2/expnu/(robs**(three/two)+a_spin)*zero
      vel(3)=0.0D0!exppsi/expnu*(-somiga)  !exppsi/expnu*(one/(robs**(three/two)+a_spin)-somiga) 

      vptl(1)= -0.1D0!velo*cos(theta)
      vptl(2)= 0.0D0!velo*sin(theta)*sin(phi_i)
      vptl(3)= 0.1D0!velo*sin(theta)*cos(phi_i)

      call lambdaqm(vptl,sin_ini,cos_ini,a_spin,e,r_ini,sign_ch,vel,lambda,q,mve,ep,kp) 

      p = 0.00001D0
      time_0 = 0.D0
      sigma_0 = 246.D0 
      r_ball = 4.D0
       
      x_c = dsqrt(r_ini*r_ini+a_spin*a_spin)*sin_ini*dcos(phi_ini*dtor) 
      y_c = dsqrt(r_ini*r_ini+a_spin*a_spin)*sin_ini*dsin(phi_ini*dtor)
      z_c = dsqrt(r_ini*r_ini+a_spin*a_spin)*cos_ini

      open(unit=15,file='./xx2.txt',status="replace")
      open(unit=16,file='./yy2.txt',status="replace")
      open(unit=17,file='./zz2.txt',status="replace")
          Do i = 0,n1-1
              the = i*pi/real(n1)
              Do j = 0,m1-1
                  phi1 = j*twopi/real(m1)
                  xp1 = x_c + r_ball*dsin(the)*dcos(phi1)
                  yp1 = y_c + r_ball*dsin(the)*dsin(phi1)     
                  zp1 = z_c + r_ball*dcos(the)          

                  write(unit=15,fmt=*)xp1
                  write(unit=16,fmt=*)yp1
                  write(unit=17,fmt=*)zp1
                  xp(i*m1+j+1) = xp1
                  yp(i*m1+j+1) = yp1
                  zp(i*m1+j+1) = zp1
              Enddo
          Enddo

      Do k = 4,1,-1
          t = zero+90.D0*k/four

          Do i = 0,n1-1
              the = i*pi/real(n1)
              Do j = 0,m1-1
                  xp1 = xp(i*m1+j+1)
                  yp1 = yp(i*m1+j+1) 
                  zp1 = zp(i*m1+j+1) 
                  r_ini = dsqrt(xp1*xp1+yp1*yp1+zp1*zp1-a2)
                  sin_ini = dsqrt( (xp1*xp1+yp1*yp1)/(xp1*xp1+yp1*yp1+zp1*zp1) )
                  cos_ini = zp1/dsqrt(xp1*xp1+yp1*yp1+zp1*zp1)
                  phy_ini = dasin( yp1/dsqrt(xp1*xp1+yp1*yp1) )

                  !write(*,*)r_ini,sin_ini,cos_ini
                  call lambdaqm(vptl,sin_ini,cos_ini,a_spin,e,r_ini,sign_ch,vel,lambda,q,mve,ep,kp) 
                   !write(*,*)lambda,q,mve,ep,kp  
                  p_t = time2p_bisection(t,kp,r_ini,sin_ini,cos_ini,&
                                            a_spin,e,lambda,q,mve,ep)     
 
                  write(*,*)'number==',k,i,j,t,p_t,kp(1) 
                  CALL YNOGKM(p_t,kp,lambda,q,mve,ep,sin_ini,cos_ini,a_spin,e,r_ini,&
                                                    radi,mu,tim,phy,sigm,cir_orbt,theta_star)
 
                  xp1 = dsqrt(radi*radi+a2)*dsqrt(one-mu*mu)*dcos(phy+phy_ini)
                  yp1 = dsqrt(radi*radi+a2)*dsqrt(one-mu*mu)*dsin(phy+phy_ini)     
                  zp1 = dsqrt(radi*radi+a2)*mu          

                  write(unit=15,fmt=*)xp1
                  write(unit=16,fmt=*)yp1
                  write(unit=17,fmt=*)zp1
              Enddo
          Enddo
      Enddo

      close(unit=15)	
      close(unit=16)
      close(unit=17)

      end


!******************************************************************
