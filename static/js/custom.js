$(document).ready( function() {
       

      
      $("#form").submit(function(e){
      
      var error = false;
      e.preventDefault();
      
      var image_files = $("#image_files").val();
      var height = $("#height").val();

      if (image_files==="" ){
          alert("no image files");
          error = true;
      }
      
      if (height===""){
          alert("please specify height");
          error = true;
      }
 
     
     
     if (!error){
         e.currentTarget.submit();
         /*
              console.log("trying to call ajax");
               $.ajax({
                 
                     data: {
                         image_files, 
                         height
                     },
                     type:"POST", 
                     url: "/predict_image"
                     
                 })
        */
     
     }
          
     
     
      });
      
      
      

	});
