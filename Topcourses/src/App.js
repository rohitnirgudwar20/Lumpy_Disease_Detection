import logo from './logo.svg';
import './App.css';
import { filterdata,apiUrl } from './data';
import Navbar from './components/Navbar';
import Filter from './components/filterData';
import Tokens from './components/Tokens';
import Spinner  from './components/Spinner';
import { toast } from 'react-toastify';
import { useEffect, useState } from 'react';
  

function App() {


  // const [courses,setcourses] =useState("something");
  const [datas, setdata] =useState(null);
  const [loading,setloading] =useState(true);
  const [category,setcategory]= useState(filterdata[0].title);

  async function getdata(){
    setloading(true);
    try{
      let response =  await fetch(apiUrl);
      let output = await response.json();
 
      setdata(output.data);
      console.log(output);

      console.log("======output.data=======");
      console.log(output.data);

      
    }

    catch{
      console.log(" Error occurs in app");
      

    }
    setloading(false);
  }

  useEffect( ()=>{
    getdata();

  },[])

  
  return (
    <div className="min-h-screen flex flex-col bg-red-700">

      <div>
      <Navbar/>
      </div>

      <div>
      <Filter filterdata={filterdata}   category={category}
        setcategory={setcategory} />
      </div>

      <div className='w-11/12 max-w-[1200px] mx-auto flex flex-wrap  justify-center
            items-center min-h-[50vh]'>
        {
          loading? (<Spinner/>):(<Tokens  courses ={datas}   category={category}/>)  
        }
      
      </div>  
    </div>
  );
}

export default App;
