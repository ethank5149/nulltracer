pro streamlines

  oldn=!D.name & set_plot,'ps'
  
 ; !p.font=0
 
 n_x=1
 
nm=19uL
ln=801ul
length = 15

 Openr,xcoor,'rayx1.txt',/Get_Lun
  Point_lun,xcoor,0
  xcoord1=fltarr(ln,nm)
  ReadF,xcoor,xcoord1
  
   Openr,ycoor,'rayz1.txt',/Get_Lun
  Point_lun,ycoor,0
  ycoord1=fltarr(ln,nm)
  ReadF,ycoor,ycoord1
 
  
  free_lun,xcoor,ycoor

  ;===============================================
  NN=100
  MM=450
  A=1.2
  B=1.05
    Lc=5.
  ratio=1;3.529
   l=16 & xxss=l*(ratio) & yyss=l
   ;xoff=(LL-xxss)/2.,yoff=(3*LL/2.-yyss)/2.,
    !p.font = 0
   device,filename='streamlines_2.ps',xsize=xxss,ysize=yyss,bits_per_pixel=8,/color,xoff=(21.5-xxss)/2.0,yoff=(30-yyss)/2.,$
   set_font='Times-Roman';, /tt_font
  ; set_font='isolatin1'
   XX=Findgen(101)
   YY=Findgen(101)
   XXX=findgen(11)*(0.1)
  rr=fltarr(100,100)*0+1
  cc=findgen(100)*(7.1416/99)
  xc=cos(findgen(100))
  yc=sin(findgen(100))
 
; loadct,30
  RRR=bytscl(findgen(256))
  GGG=bytscl(findgen(256))
  BBB=bytscl(findgen(256))
  RRR[248:255]=[0,  0,255,255,0, 0,255,255]
  GGG[248:255]=[255,0,0,  255,0,255,0,255]
  BBB[248:255]=[0,  0,0,  0  ,255,255,255,255]
  ;248=green,249=black,250=red,251=yellow,252=blue,253=cyan,254=magenta,255=white
  TVLCT,RRR,GGG,BBB
  
  ;CCol=[248,249,250,251,252,253,254]
  ;CCol=252
  
 ;plot,[0,1.5],[0,1.1],pos=[(xxss/(2*Lc))*1000,(yyss/(2*Lc))*1000,$
  ;(xxss*(2*Lc-1)/(2*Lc))*1000,(yyss*(2*Lc-1)/(2*Lc))*1000],/noerase,/nodata,$
  ;         /device,/ynozero,charsize=1,charthick=3;,yticklen=0.001;,yminor=5  
	xlen=0.5
	ylen=0.5
	xslitlen=0.
	yslitlen=0.
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
		If ny eq 0 and nx eq 0 then begin
		    for i=0,nm-1 do begin
		    	oplot,xcoord1(*,i),ycoord1(*,i),color=249 
		    	oplot,xcoord1(*,i),-ycoord1(*,i),color=249 
		    	oplot,-xcoord1(*,i),ycoord1(*,i),color=249 
		    	oplot,-xcoord1(*,i),-ycoord1(*,i),color=249 
		    endfor
		endif  
;************************************************************
                ;this part to draw a out circle of the disk
                aspin = 0.998  
	        rhori=1.+sqrt(1.-aspin*aspin)

                delta=360*!dtor/200.	
                aaa=fltarr(200) & disklx=aaa & diskly=aaa & disklz=aaa
                thetax=aaa & j=indgen(211)
                thetax=j*delta
                disklx=rhori*cos(thetax)
                diskly=rhori*sin(thetax)
	
                plots,disklx,diskly,color=1b,thick=8b
                diskl=15.
                disklx=diskl*cos(thetax)
                diskly=diskl*sin(thetax)
                plots,disklx,diskly,color=1b,thick=1b 
                plots,[-7,-rhori],[0,0],color=1b,thick=4b 
                plots,[rhori,7],[0,0],color=1b,thick=4b 
                ;print,thetax,j,delta
;*******************************************************************************************************
		axis,xaxis=0,xticks=1,xtickname=replicate(' ',2),xminor=1,xrange=[0,15],xstyle=1
		axis,xaxis=1,xticks=1,xtickname=replicate(' ',2),xminor=1,xrange=[0,15],xstyle=1
		axis,yaxis=0,yticks=1,yminor=1,yrange=[0,15],ystyle=1,ytickname=replicate(' ',2);,ytickv=[0,0.5,1]
	 	axis,yaxis=1,yticks=1,yminor=1,yrange=[0,15],ystyle=1,ytickname=replicate(' ',2);,ytickv=[0,0.5,1]
		;oplot,fluxx(*,ny*mx+nx),flux(*,ny*mx+nx),thick=2,color=249;,$;,psym=-4

		colors=1
		chsize=0.8
		chth=2
		tickth=2
		alp1=-15
		alp2=15
		beta1=-15
		beta2=15
		xminor =5 
                yminor =5
                xticks =6
                yticks =6
                xtickslen=0.01
                ytickslen=0.01
          if (ny eq 0) then begin
            ;******************************************************** 
                if nx eq 0 then begin
		    axis,xaxis=0,xticks=xticks,xminor=xminor,xrange=[alp1,alp2],xstyle=1,charsize=chsize,charthick=chth,font=0,$
		    xthick=tickth,xtitle=textoidl('R [GM/c^2]'),color=colors,xticklen=xtickslen;,xtickname=replicate(' ',11);,$
			;xtickname=['-3','-1','1','3','5','7'],
                endif
                if nx ne 0 then begin 
		    axis,xaxis=0,xticks=xticks,xminor=xminor,xrange=[alp1,alp2],xstyle=1,xtickname=replicate(' ',11),font=0,$
		    charsize=chsize,charthick=chth,xthick=tickth,color=colors,xticklen=xtickslen
                endif

		axis,xaxis=1,xticks=xticks,xminor=xminor,xrange=[alp1,alp2],xstyle=1,xtickname=replicate(' ',11),font=0,$
		charsize=chsize,charthick=chth,xthick=tickth,color=colors,xticklen=xtickslen

		axis,yaxis=0,yticks=yticks,yminor=yminor,yrange=[beta1,beta2],ystyle=1,charsize=chsize,charthick=chth,font=0,$
		ythick=tickth,ytitle=textoidl('z [GM/c^2]'),color=colors,yticklen=ytickslen;ytickname=replicate(' ',11),$
			;ytickname=['-6','-4','-2','0','2','4','6'],
		;,ytickv=[0,0.5,1]

	 	axis,yaxis=1,yticks=yticks,yminor=yminor,yrange=[beta1,beta2],ystyle=1,ytickname=replicate(' ',11),font=0,$
		charsize=chsize,charthick=chth,ythick=tickth,color=colors,yticklen=ytickslen;
            ;*******************************************************
          endif 
          if (nx eq 0) and (ny ne 0) then begin 
            ;******************************************************** 
		;axis,xaxis=0,xticks=xticks,xminor=xminor,xrange=[alp1,alp2],xstyle=1,charsize=chsize,charthick=chth,font=0,$
		;xthick=tickth,xtitle=textoidl('X [GM/c^2]'),color=colors,xticklen=xtickslen;,xtickname=replicate(' ',11);,$
			;xtickname=['-4','-2','0','2','4','6','8'],

		axis,xaxis=0,xticks=xticks,xminor=xminor,xrange=[alp1,alp2],xstyle=1,xtickname=replicate(' ',11),font=0,$
		charsize=chsize,charthick=chth,xthick=tickth,color=colors,xticklen=xtickslen 

		axis,xaxis=1,xticks=xticks,xminor=xminor,xrange=[alp1,alp2],xstyle=1,xtickname=replicate(' ',11),font=0,$
		charsize=chsize,charthick=chth,xthick=tickth,color=colors,xticklen=xtickslen

		axis,yaxis=0,yticks=yticks,yminor=yminor,yrange=[beta1,beta2],ystyle=1,charsize=chsize,charthick=chth,font=0,$
		ythick=tickth,ytitle=textoidl('Y [GM/c^2]'),color=colors,yticklen=ytickslen,ytickname=replicate(' ',1);,$
			;ytickname=['-6','-4','-2','0','2','4','6'],
		;,ytickv=[0,0.5,1]

	 	axis,yaxis=1,yticks=yticks,yminor=yminor,yrange=[beta1,beta2],ystyle=1,ytickname=replicate(' ',11),font=0,$
		charsize=chsize,charthick=chth,ythick=tickth,color=colors,yticklen=ytickslen;
            ;*******************************************************
            endif
	endfor
   endfor 	
   ;plot,[0,1],[0,1],pos=[0,0,1,1],$
   ;	/noerase,/device,/ynozero,/normal,xstyle=4+1,ystyle=4+1,/nodata
  ;axis,xaxis=0,xtitle=textoidl('\nu/\nu_{em}'),xticks=1,xminor=1,xtickname=replicate(' ',2);,xstyle=4
  ;axis,yaxis=0,ytitle='Observed flux (arbitrary units)',yticks=1,yminor=1,ytickname=replicate(' ',2);,ystyle=4
        ;xyouts,0.47,0.04,textoidl('\nu/\nu_{em}'),charsize=1.2  ;,/device,/normal,
	;xyouts,0.07,0.25,'Observed flux (arbitrary units)',orientation=90,charsize=1.5
   device,/close 
    
   set_plot,oldn
   
end
