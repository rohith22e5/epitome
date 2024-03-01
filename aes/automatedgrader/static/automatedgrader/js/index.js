
document.addEventListener('DOMContentLoaded', () => {
   let toggleBtn = document.getElementById('toggle-btn');
   let body = document.body;
   
   let profile = document.querySelector('.header .flex .profile');
   
   document.querySelector('#user-btn').onclick = () =>{
      profile.classList.toggle('active');
      search.classList.remove('active');
   }
   
   let search = document.querySelector('.header .flex .search-form');
   
   document.querySelector('#search-btn').onclick = () =>{
      search.classList.toggle('active');
      profile.classList.remove('active');
   }
   
   let sideBar = document.querySelector('.side-bar');
   
   document.querySelector('#menu-btn').onclick = () =>{
      sideBar.classList.toggle('active');
      body.classList.toggle('active');
   }
   
   document.querySelector('#close-btn').onclick = () =>{
      sideBar.classList.remove('active');
      body.classList.remove('active');
   }
   
   window.onscroll = () =>{
      profile.classList.remove('active');
      search.classList.remove('active');
   
      if(window.innerWidth < 1200){
         sideBar.classList.remove('active');
         body.classList.remove('active');
      }
   }

   const form = document.querySelector('#inputform');
   form.onsubmit = function(e) {
      e.preventDefault(); // Prevent the default form submission
      const formData = new FormData(form);
      
      fetch("{% url 'process_essay' %}", {
        method: "POST",
        body: formData,
        headers: {
          'X-CSRFToken': form.querySelector('[name=csrfmiddlewaretoken]').value,
        },
      })
      .then(response => response.json())
      .then(data => {
        console.log(data); // Handle the JSON response here
        // For example, update the DOM with the word count or feedback
      })
      .catch(error => console.error('Error:', error));
    };
  });



