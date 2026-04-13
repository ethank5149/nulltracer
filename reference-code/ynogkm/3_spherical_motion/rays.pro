pro  rays

theta0=0.3
phi0=0
WINSIZE=20
bkg='wwhite'
backg='ny'
nm=1uL
ln=15001ul
nmstart=0ul
lee=1ul
lbb=0ul
;@nlff_block

if not keyword_set(bkg) then bkg='white' else bkg=bkg(0)

 Openr,xcoor,'rayx1.txt',/Get_Lun
  Point_lun,xcoor,0
  xcoord=fltarr(ln,nm)
  ReadF,xcoor,xcoord
  
   Openr,ycoor,'rayy1.txt',/Get_Lun
  Point_lun,ycoor,0
  ycoord=fltarr(ln,nm)
  ReadF,ycoor,ycoord
  
   Openr,zcoor,'rayz1.txt',/Get_Lun
  Point_lun,zcoor,0
  zcoord=fltarr(ln,nm)
  ReadF,zcoor,zcoord
 
 
  free_lun,xcoor,ycoor,zcoor; 
 

  length=2.5
xmin=-length
xmax=length
ymin=-length
ymax=length
zmin=-length
zmax=length


if(phi0 lt 0) then begin
   phi1=phi0+360
   phi=!dtor*phi1
endif else begin
   phi=phi0*!dtor
endelse
theta=theta0*!dtor

st=sin(theta) & ct=cos(theta)
sp=sin(phi)   & cp=cos(phi)

ok=0
aga_flag=0
addnum=''

oldn=!D.name & set_plot,'ps'
 
loadct,33  ;  loadct,3 also looks nice too
 
    l=16  & xxss=l & yyss=l
    !p.font = 0
    device,filename='./sph_motion_2.ps',color=1,xsize=xxss,ysize=yyss,$
         xoff=(21.-xxss)/2. , yoff=(21-yyss)/2.,bits_per_pixel=8,set_font='Times-Roman';decomposed=0

	xlen=0.7
	ylen=0.7
	xslitlen=0.04
	yslitlen=0.04
	mx=1 ;figure numbers on horizon direction
	my=1 ;figure numbers on vertical direction.
	deltax=[(xlen-xslitlen)/mx+xslitlen,0,(xlen-xslitlen)/mx+xslitlen,0]
	deltay=[0,(ylen-yslitlen)/my+yslitlen,0,(ylen-yslitlen)/my+yslitlen]
	deltxy=[xlen/mx,ylen/my,xlen/mx,ylen/my]
	pos_llft=[(1.-xlen)/2.,(1.-ylen)/2.,(1.-xlen)/2.+(xlen-xslitlen)/mx,(1.-ylen)/2.+(ylen-yslitlen)/my]
   for ny=0,my-1 do begin	
     	for nx=0,mx-1 do begin	
		plot,[-length,length],[-length,length],pos=[pos_llft+deltax*nx+deltay*ny],xrange=[-length,length],yrange=[-length,length],$
		/noerase,/device,/ynozero,/normal,xstyle=4+1,ystyle=4+1,charsize=0.5,xtickv=[0,0.5,1,1.5],$
		xticks=10,xminor=10,xtickname=replicate(' ',10),/nodata 
;^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
 
                loadct,3
                tvlct,re,gr,bl,/get
                re(249:255)=[0b,255b,0b,0b,255b,0b,255b]
                gr(249:255)=[0b,0b,255b,0b,255b,255b,255b]
                bl(249:255)=[0b,0b,0b,255b,0b,255b,255b]
                tvlct,re,gr,bl
 
 

    ;this part to draw a out circle of the disk 
	diskl=1.35
	delta=360*!dtor/200.	
	aaa=fltarr(200) & disklx=aaa & diskly=aaa & disklz=aaa
	thetax=aaa & j=indgen(211)
	thetax=j*delta
	disklx=diskl*cos(thetax)
	diskly=diskl*sin(thetax)
	
	;plots,disklx,diskly,color=1b,thick=8b 
  
        for i=nmstart,nm-1 do begin
  
            x1=xcoord[i*ln+lbb:(i+1)*ln-lee]*cp-ycoord[i*ln+lbb:(i+1)*ln-lee]*sp
            y1=-xcoord[i*ln+lbb:(i+1)*ln-lee]*sp+ycoord[i*ln+lbb:(i+1)*ln-lee]*cp
            z1=zcoord[i*ln+lbb:(i+1)*ln-lee]
  
            x2=x1 
            y2=y1*ct-z1*st
            z2=y1*st+z1*ct
            col=249b
            oplot,x2,z2,col=col,thick=1.2  
        endfor
;******************************************************** 
		colors=1
		chsize=0.7
		chth=2
		tickth=2
		alp1=-length
		alp2=length
		beta1=-length
		beta2=length
		xminor =2 
                yminor =2
                xticks =10
                yticks =10
                xtickslen=0.01
                ytickslen=0.01
		axis,xaxis=0,xticks=xticks,xminor=xminor,xrange=[alp1,alp2],xstyle=1,charsize=chsize,charthick=chth,font=0,$
		xthick=tickth,xtitle=textoidl('X [GM/c^2]'),color=colors,xticklen=xtickslen;,xtickname=replicate(' ',11);,$
			;xtickname=['-4','-2','0','2','4','6','8'],

		axis,xaxis=1,xticks=xticks,xminor=xminor,xrange=[alp1,alp2],xstyle=1,xtickname=replicate(' ',11),font=0,$
		charsize=chsize,charthick=chth,xthick=tickth,color=colors,xticklen=xtickslen

		axis,yaxis=0,yticks=yticks,yminor=yminor,yrange=[beta1,beta2],ystyle=1,charsize=chsize,charthick=chth,font=0,$
		ythick=tickth,ytitle=textoidl('Y [GM/c^2]'),color=colors,yticklen=ytickslen;ytickname=replicate(' ',11),$
			;ytickname=['-6','-4','-2','0','2','4','6'],
		;,ytickv=[0,0.5,1]

	 	axis,yaxis=1,yticks=yticks,yminor=yminor,yrange=[beta1,beta2],ystyle=1,ytickname=replicate(' ',11),font=0,$
		charsize=chsize,charthick=chth,ythick=tickth,color=colors,yticklen=ytickslen;
            ;******************************************************* 
	endfor
   endfor 

   device,/close 
 
   set_plot,oldn
 
 end
