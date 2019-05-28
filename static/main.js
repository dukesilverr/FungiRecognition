$('document').ready(function(){
    var input = document.getElementById( 'file-input' );
    var label	 = input.nextElementSibling,
    labelVal = label.innerHTML;
        input.addEventListener( 'change', function( e )
            {
                var fileName = '';
                if( this.files && this.files.length > 1 ) {
                    if(window.location.pathname.includes("lt")) {
                        fileName= this.files.length + " pasirinkti failai"
                    } else {
                    	fileName = ( this.getAttribute( 'data-multiple-caption' ) || '' ).replace( '{count}', this.files.length );
		 	}
		} else {
                    fileName = e.target.value.split( '\\' ).pop();
		}
                if( fileName ) {
                    label.querySelector( 'span' ).innerHTML = fileName;
                } else {
                    label.innerHTML = labelVal;
		}
        });
});

