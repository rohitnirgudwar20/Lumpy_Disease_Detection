import React, { useState } from 'react'
import Token from './Token';

const Tokens = (props) => {
  let courses = props.courses;
  const [likedCourse, setLikedCourse] = useState([]);
  let   category=props.category;
  
  function getcourses(){
    let allcourses = [];


    if(category=='All'){
        Object.values(courses).forEach( coursetitle =>{
            coursetitle.forEach( coursedata =>{
                    allcourses.push(coursedata);
                } )
        } )

        return allcourses;
        
    }
    else{
        return courses[category];
    }
    
    
  }
   

  return ( 
    <div  className='flex flex-wrap  justify-center gap-4 mb-4'>
        {
            getcourses().map((course) =>(
                   <Token key={course.id} course={course}  likedCourses={likedCourse} setLikedCourses={setLikedCourse}/>
            ))
            
        }
    </div>
  )
}

export default Tokens